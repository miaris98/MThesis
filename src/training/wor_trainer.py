"""World on Rails (WoR) Distillation Trainer.

Trains the sensorimotor vision policy to predict optimal Q-values and waypoints
using differential learning rates, PyTorch AMP (Automatic Mixed Precision),
and telemetry tracking (MLflow + TensorBoard + per-epoch CSV, matching the
PPO/SAC trainers' logging stack).
"""
from typing import Dict, Optional, Tuple
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy
from src.training.wor_dataset import create_wor_dataloader
from src.logging.csv_logger import CSVTelemetryLogger
from src.logging.experiment_logger import ExperimentLogger
from src.logging.hardware_monitor import HardwareMonitor


try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    from torch.amp import GradScaler, autocast


class WorldOnRailsTrainer:
    """
    Trainer for World on Rails Policy Distillation.
    """

    #: Per-epoch telemetry schema (mirrors the PPO/SAC trainers' CSV+MLflow+TensorBoard
    #: stack so offline WoR runs are inspectable/comparable the same way).
    TELEMETRY_FIELDS = [
        "epoch", "num_batches", "wall_time_s", "epoch_time_sec", "samples_per_sec",
        "data_wait_sec", "compute_sec",
        "total_loss", "q_loss", "wp_loss", "wp_ade_m", "wp_fde_m",
        "wp_lateral_error_m", "wp_longitudinal_error_m",
        "lr_backbone", "lr_heads", "grad_norm", "is_best",
        "gpu_mem_used_mb", "gpu_mem_pct", "sys_cpu_pct", "sys_ram_used_gb"
    ]

    def __init__(
        self,
        model: WorldOnRailsPolicy,
        data_dir: str,
        val_data_dir: Optional[str] = None,
        save_dir: str = "checkpoints/wor",
        lr_backbone: float = 1e-4,
        lr_heads: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        wp_loss_weight: float = 1.0,
        q_loss_weight: float = 0.0,
        lateral_loss_weight: float = 3.0,
        synthetic_samples: int = 0,
        cache_decoded: bool = True,
        compile_model: bool = False,
        route_points: int = 4,
        experiment_name: str = "WoR_Offline_Training",
        use_mlflow: bool = True,
        mlflow_port: int = 10100
    ):
        # channels_last suits conv+AMP on tensor cores, and costs nothing to feed:
        # frames arrive from the dataset as uint8 HWC, which permutes to NCHW with
        # channels_last layout without a copy. Checkpoints are unaffected - memory
        # format isn't part of state_dict - so eval can still load these weights into
        # a contiguous model.
        self.model = model.to(device, memory_format=torch.channels_last)
        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.save_dir = save_dir
        self.device = device
        self.use_amp = use_amp and (device == "cuda")
        self.wp_loss_weight = wp_loss_weight
        # Datasets without precomputed Q-values (e.g. PDM-Lite) leave target_q at
        # zero, so q_loss_weight defaults to 0 to avoid supervising toward zero.
        self.q_loss_weight = q_loss_weight
        # Waypoints are (x_forward, y_lateral). Forward displacement is typically
        # several meters per waypoint while lateral offset - the ONLY component the
        # PID controller's steering comes from (PIDController.control_from_waypoints
        # reads aim_point[1]) - is often under a meter. A flat L1 loss over both axes
        # lets the large-magnitude x term dominate the gradient, so the network can
        # minimize loss mostly by nailing forward distance while barely fitting y -
        # producing a policy that accelerates fine but steers close to zero. Weight
        # the lateral term up to correct for that scale mismatch.
        self.lateral_loss_weight = lateral_loss_weight
        self.train_start_time = time.time()

        os.makedirs(save_dir, exist_ok=True)

        # MLflow + TensorBoard (same unified logger the PPO/SAC trainers use).
        self.logger = ExperimentLogger(
            save_dir, checkpoint_dir=save_dir,
            experiment_name=experiment_name, use_mlflow=use_mlflow, mlflow_port=mlflow_port
        )
        self.logger.log_params({
            "data_dir": data_dir, "backbone": model.encoder.backbone_name,
            "lr_backbone": lr_backbone, "lr_heads": lr_heads, "batch_size": batch_size,
            "wp_loss_weight": wp_loss_weight, "q_loss_weight": q_loss_weight,
            "lateral_loss_weight": lateral_loss_weight
        })

        # Per-epoch CSV telemetry.
        self.csv_logger = CSVTelemetryLogger(
            os.path.join(save_dir, "wor_training_telemetry.csv"), fieldnames=self.TELEMETRY_FIELDS
        )

        # 1. DataLoaders
        self.train_loader = create_wor_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=True,
            synthetic_samples=synthetic_samples,
            cache_decoded=cache_decoded,
            route_points=route_points
        )
        if len(self.train_loader.dataset) == 0 or getattr(self.train_loader.dataset, "is_synthetic", False):
            print(f"[Warning] Training on SYNTHETIC data - no real frames were indexed under {data_dir}.")

        # 2. Parameter Groups with Differential Learning Rate
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_heads}
        ]

        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)
        try:
            self.scaler = GradScaler(enabled=self.use_amp)
        except Exception:
            self.scaler = GradScaler()

        # Compiled last, after the optimizer already holds references to the real
        # parameters - torch.compile wraps the module without copying params, so the
        # optimizer stays correct. Checkpoints stay loadable too: compiling adds an
        # "_orig_mod." prefix to state_dict keys, which wor_loader already strips.
        # Costs a one-off graph compile on the first epoch, so it only pays off over
        # a long run.
        if compile_model:
            if not hasattr(torch, "compile"):
                print("[Warning] torch.compile unavailable on this torch version - running uncompiled.")
            else:
                try:
                    self.model = torch.compile(self.model)
                    print("--> torch.compile enabled (first epoch includes one-off compilation).")
                except Exception as e:
                    print(f"[Warning] torch.compile failed ({e}) - running uncompiled.")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full training epoch."""
        self.model.train()
        total_loss_accum = 0.0
        q_loss_accum = 0.0
        wp_loss_accum = 0.0
        ade_accum = 0.0
        fde_accum = 0.0
        lateral_err_accum = 0.0
        longitudinal_err_accum = 0.0
        grad_norm_accum = 0.0
        num_batches = 0
        num_samples = 0
        # Split the epoch into "blocked waiting for the dataloader" vs "actually
        # computing" so the next optimization targets whichever one dominates,
        # instead of guessing (batch size was raised 8x once for no speedup at all,
        # because the pipeline was data-bound the whole time).
        data_wait = 0.0
        compute_time = 0.0
        start_time = time.time()
        t_batch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            data_wait += time.time() - t_batch_start
            t_compute_start = time.time()

            # uint8 HWC -> float NCHW in [0,1], done on the GPU: a quarter of the
            # PCIe traffic of sending float32, and the permute lands in channels_last
            # without a copy since the source is already HWC.
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            rgb = rgb.permute(0, 3, 1, 2).float().div_(255.0)
            speed = batch["speed"].to(self.device, non_blocking=True)
            command = batch["command"].to(self.device, non_blocking=True)
            route = batch["route"].to(self.device, non_blocking=True)
            target_q = batch["target_q"].to(self.device, non_blocking=True)
            target_wp = batch["target_waypoints"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                out = self.model(rgb, speed, command, route)

                # 1. Q-value distillation loss (MSE on selected rail Q-values)
                pred_q = out["selected_rail_q"]
                loss_q = F.mse_loss(pred_q, target_q)

                # 2. Waypoint imitation loss, split per-axis so the lateral (steering)
                # component can be weighted independently of the larger-magnitude
                # forward component (see lateral_loss_weight in __init__).
                pred_wp = out["selected_waypoints"]
                loss_wp_x = F.l1_loss(pred_wp[..., 0], target_wp[..., 0])
                loss_wp_y = F.l1_loss(pred_wp[..., 1], target_wp[..., 1])
                loss_wp = loss_wp_x + self.lateral_loss_weight * loss_wp_y

                total_loss = self.q_loss_weight * loss_q + self.wp_loss_weight * loss_wp

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            # Average/Final Displacement Error (meters) - interpretable trajectory-quality
            # metrics on top of the raw L1 waypoint loss. Also track the unweighted
            # per-axis error directly so a lateral/longitudinal imbalance (the
            # near-zero-steering failure mode) is visible in telemetry even though the
            # loss above weights the axes unevenly on purpose.
            with torch.no_grad():
                per_point_dist = torch.norm(pred_wp.float() - target_wp.float(), dim=-1)  # (B, 5)
                ade = per_point_dist.mean().item()
                fde = per_point_dist[:, -1].mean().item()
                lateral_err = loss_wp_y.item()
                longitudinal_err = loss_wp_x.item()

            total_loss_accum += total_loss.item()
            q_loss_accum += loss_q.item()
            wp_loss_accum += loss_wp.item()
            ade_accum += ade
            fde_accum += fde
            lateral_err_accum += lateral_err
            longitudinal_err_accum += longitudinal_err
            grad_norm_accum += float(grad_norm)
            num_batches += 1
            num_samples += rgb.shape[0]

            # CUDA work is async, so the compute window has to be closed on a sync or
            # its cost would silently land in the next iteration's data-wait bucket.
            if self.device == "cuda":
                torch.cuda.synchronize()
            compute_time += time.time() - t_compute_start
            t_batch_start = time.time()

        avg_loss = total_loss_accum / max(1, num_batches)
        avg_q_loss = q_loss_accum / max(1, num_batches)
        avg_wp_loss = wp_loss_accum / max(1, num_batches)
        avg_ade = ade_accum / max(1, num_batches)
        avg_fde = fde_accum / max(1, num_batches)
        avg_lateral_err = lateral_err_accum / max(1, num_batches)
        avg_longitudinal_err = longitudinal_err_accum / max(1, num_batches)
        avg_grad_norm = grad_norm_accum / max(1, num_batches)
        elapsed = time.time() - start_time
        samples_per_sec = num_samples / max(1e-6, elapsed)

        lr_b = self.optimizer.param_groups[0]["lr"]
        lr_h = self.optimizer.param_groups[1]["lr"]

        return {
            "epoch": epoch,
            "num_batches": num_batches,
            "total_loss": avg_loss,
            "q_loss": avg_q_loss,
            "wp_loss": avg_wp_loss,
            "wp_ade_m": avg_ade,
            "wp_fde_m": avg_fde,
            "wp_lateral_error_m": avg_lateral_err,
            "wp_longitudinal_error_m": avg_longitudinal_err,
            "grad_norm": avg_grad_norm,
            "lr_backbone": lr_b,
            "lr_heads": lr_h,
            "samples_per_sec": samples_per_sec,
            "data_wait_sec": data_wait,
            "compute_sec": compute_time,
            "time": elapsed
        }

    def train(self, num_epochs: int = 50, save_freq: int = 5):
        """Runs the full distillation training loop with checkpointing."""
        print(f"--> Starting World on Rails Distillation Training for {num_epochs} epochs on {self.device.upper()}...")
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        best_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(epoch)
            scheduler.step()
            is_best = metrics["total_loss"] < best_loss

            print(
                f"[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"Total Loss: {metrics['total_loss']:.4f} | "
                f"Q Loss: {metrics['q_loss']:.4f} | "
                f"WP Loss: {metrics['wp_loss']:.4f} | "
                f"ADE: {metrics['wp_ade_m']:.3f}m | FDE: {metrics['wp_fde_m']:.3f}m | "
                f"Lat Err: {metrics['wp_lateral_error_m']:.3f}m | Lon Err: {metrics['wp_longitudinal_error_m']:.3f}m | "
                f"Samples/s: {metrics['samples_per_sec']:.1f} | Time: {metrics['time']:.2f}s "
                f"(data {metrics['data_wait_sec']:.1f}s / compute {metrics['compute_sec']:.1f}s)"
            )

            hw = HardwareMonitor.get_metrics()
            for tag in ("total_loss", "q_loss", "wp_loss", "wp_ade_m", "wp_fde_m",
                        "wp_lateral_error_m", "wp_longitudinal_error_m",
                        "grad_norm", "lr_backbone", "lr_heads", "samples_per_sec",
                        "data_wait_sec", "compute_sec"):
                self.logger.add_scalar(f"wor/{tag}", metrics[tag], epoch)

            self.csv_logger.log_step({
                "epoch": epoch, "num_batches": metrics["num_batches"],
                "wall_time_s": round(time.time() - self.train_start_time, 2),
                "epoch_time_sec": round(metrics["time"], 2),
                "samples_per_sec": round(metrics["samples_per_sec"], 1),
                "data_wait_sec": round(metrics["data_wait_sec"], 2),
                "compute_sec": round(metrics["compute_sec"], 2),
                "total_loss": round(metrics["total_loss"], 5),
                "q_loss": round(metrics["q_loss"], 5),
                "wp_loss": round(metrics["wp_loss"], 5),
                "wp_ade_m": round(metrics["wp_ade_m"], 4),
                "wp_fde_m": round(metrics["wp_fde_m"], 4),
                "wp_lateral_error_m": round(metrics["wp_lateral_error_m"], 4),
                "wp_longitudinal_error_m": round(metrics["wp_longitudinal_error_m"], 4),
                "lr_backbone": f"{metrics['lr_backbone']:.2e}",
                "lr_heads": f"{metrics['lr_heads']:.2e}",
                "grad_norm": round(metrics["grad_norm"], 4),
                "is_best": is_best,
                "gpu_mem_used_mb": hw["gpu_mem_used_mb"], "gpu_mem_pct": hw["gpu_mem_pct"],
                "sys_cpu_pct": hw["sys_cpu_pct"], "sys_ram_used_gb": hw["sys_ram_used_gb"]
            })
            self.csv_logger.flush()

            # Save latest checkpoint
            torch.save({
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics
            }, os.path.join(self.save_dir, "latest_model.pth"))

            # Save best checkpoint
            if metrics["total_loss"] < best_loss:
                best_loss = metrics["total_loss"]
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "metrics": metrics
                }, os.path.join(self.save_dir, "best_model.pth"))
                print(f"★ Saved new best model checkpoint (Loss: {best_loss:.4f})")

            if epoch % save_freq == 0:
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                }, os.path.join(self.save_dir, f"model_epoch_{epoch:03d}.pth"))

        self.csv_logger.close()
        if os.path.exists(self.csv_logger.filepath):
            self.logger.log_artifact(self.csv_logger.filepath)
        self.logger.close()
        print(f"✓ World on Rails Training completed! Checkpoints saved to: {self.save_dir}")
