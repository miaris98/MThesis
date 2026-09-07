"""World on Rails (WoR) Distillation Trainer.

Trains the sensorimotor vision policy to predict optimal Q-values and waypoints
using differential learning rates, PyTorch AMP (Automatic Mixed Precision),
and telemetry tracking (MLflow + TensorBoard + per-epoch CSV, matching the
PPO/SAC trainers' logging stack).

The objective and the held-out pass live in src/training/wor_eval.py; the optimizer
groups and LR schedule live in src/training/wor_optim.py.
"""
from typing import Dict, Optional
import math
import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy
from src.training.wor_dataset import create_wor_train_val_dataloaders
from src.training.wor_eval import run_validation, to_device_batch, waypoint_losses
from src.training.wor_checkpoint import CheckpointWriter, to_cpu
from src.training.wor_optim import build_optimizer, build_warmup_cosine_scheduler
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
        # Held-out metrics, blank when no validation set is configured. Training loss
        # alone cannot tell a model that generalises from one that has memorised, and
        # `is_best` used to be selected on it.
        "val_loss", "val_wp_ade_m", "val_wp_fde_m",
        "val_wp_lateral_error_m", "val_wp_longitudinal_error_m", "val_time_sec",
        "lr_backbone", "lr_heads", "grad_norm",
        # grad_norm is a mean over finite batches only. A single non-finite batch used
        # to poison the whole epoch's mean to inf, which is what made the periodic
        # `inf` rows in the first telemetry files unreadable. The AMP GradScaler
        # deliberately produces those (it overshoots the loss scale, skips the step and
        # backs off), so their *count* is the useful signal, not their magnitude.
        "nonfinite_grad_batches", "clipped_grad_batches", "grad_clip", "seed",
        "is_best", "save_sec",
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
        grad_clip: float = 5.0,
        warmup_frac: float = 0.05,
        decay_gates_and_norms: bool = False,
        val_split: float = 0.0,
        split_seed: int = 0,
        seed: Optional[int] = None,
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
        # Global-norm clip threshold. The default of 5.0 was chosen against the CNN
        # head, whose epoch-mean gradient norm falls under it around epoch 17 and so
        # spends the back half of a run unclipped. The 106M-parameter Qwen trunk sat
        # above it on every single epoch of a 50-epoch run, i.e. its updates were
        # scaled down throughout while its competitor's were not - an asymmetry that
        # belongs to the threshold, not to either architecture. Expose it so the two
        # can be given comparable effective step sizes.
        self.grad_clip = grad_clip
        self.warmup_frac = warmup_frac
        # Recorded per row so a telemetry CSV alone identifies which run it came from.
        self.seed = seed
        self.scheduler = None
        # Architecture geometry that a checkpoint cannot be reconstructed from by
        # shape alone, recorded so load_wor_model rebuilds it rather than guessing.
        self.model_config = {
            "backbone": model.encoder.backbone_name,
            "route_points": getattr(model, "route_points", route_points),
            "seed": seed,
            "vision_grid": getattr(model, "vision_grid", None),
            "pool_vision": getattr(model, "pool_vision", None),
            "num_vision_tokens": getattr(model, "num_vision_tokens", None)
        }
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
            "lateral_loss_weight": lateral_loss_weight,
            "grad_clip": grad_clip, "warmup_frac": warmup_frac,
            "decay_gates_and_norms": decay_gates_and_norms,
            "val_split": val_split, "val_data_dir": val_data_dir or "", "seed": seed,
            "vision_grid": getattr(model, "vision_grid", ""),
            "pool_vision": getattr(model, "pool_vision", "")
        })

        # Per-epoch CSV telemetry.
        self.csv_logger = CSVTelemetryLogger(
            os.path.join(save_dir, "wor_training_telemetry.csv"), fieldnames=self.TELEMETRY_FIELDS
        )

        # 1. DataLoaders
        self.train_loader, self.val_loader = create_wor_train_val_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            synthetic_samples=synthetic_samples,
            cache_decoded=cache_decoded,
            route_points=route_points,
            val_data_dir=val_data_dir,
            val_split=val_split,
            split_seed=split_seed,
            seed=seed
        )
        base_ds = getattr(self.train_loader.dataset, "dataset", self.train_loader.dataset)
        if len(self.train_loader.dataset) == 0 or getattr(base_ds, "is_synthetic", False):
            print(f"[Warning] Training on SYNTHETIC data - no real frames were indexed under {data_dir}.")
        if self.val_loader is None:
            print("[Warning] No validation set: 'best' will be selected on TRAINING loss, "
                  "which cannot detect overfitting. Pass --val_split or --val_data_dir.")

        # 2. Optimizer: differential LR for backbone vs heads, and a no-decay group for
        # gates, norms, embeddings and learned tokens. See src/training/wor_optim.py.
        self.lr_heads = lr_heads
        self.optimizer = build_optimizer(
            self.model, lr_backbone, lr_heads, weight_decay, decay_gates_and_norms
        )
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

        # 3. Checkpoints - see src/training/wor_checkpoint.py for the frozen-backbone
        # split and the snapshot-before-thread rule.
        self.ckpt = CheckpointWriter(
            self.model, save_dir, bool(getattr(model.encoder, "freeze_backbone", False))
        )

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
        grad_norm_batches = 0
        nonfinite_batches = 0
        clipped_batches = 0
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

            rgb, speed, command, route, target_q, target_wp = to_device_batch(batch, self.device)

            self.optimizer.zero_grad()

            # One shared definition of the objective for training and validation - see
            # src/training/wor_eval.py.
            with autocast(enabled=self.use_amp):
                out = self.model(rgb, speed, command, route)
                losses = waypoint_losses(
                    out, target_wp, target_q,
                    self.wp_loss_weight, self.q_loss_weight, self.lateral_loss_weight
                )
                total_loss = losses["total"]

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            # The LR schedule advances per optimizer step, not per epoch, because
            # warmup over ~5% of a 50-epoch run is 2.5 epochs - far too coarse to
            # resolve at epoch granularity, and warmup is exactly the phase a deep
            # pre-norm transformer is most sensitive to.
            if self.scheduler is not None:
                self.scheduler.step()

            gn = float(grad_norm)
            if math.isfinite(gn):
                grad_norm_accum += gn
                grad_norm_batches += 1
                if gn > self.grad_clip:
                    clipped_batches += 1
            else:
                # Under AMP this is routine: the GradScaler raises its scale every
                # growth_interval (2000) steps until the gradients overflow, then skips
                # the step and halves it. Counted rather than averaged, because a single
                # inf makes the epoch mean inf and destroys the column.
                nonfinite_batches += 1

            # ADE/FDE are interpretable trajectory-quality metrics on top of the raw L1
            # waypoint loss; the unweighted per-axis errors are tracked separately so a
            # lateral/longitudinal imbalance (the near-zero-steering failure mode) stays
            # visible even though the loss weights the axes unevenly on purpose.
            total_loss_accum += total_loss.item()
            q_loss_accum += losses["q"].item()
            wp_loss_accum += losses["wp"].item()
            ade_accum += losses["ade"].item()
            fde_accum += losses["fde"].item()
            lateral_err_accum += losses["lateral"].item()
            longitudinal_err_accum += losses["longitudinal"].item()
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
        avg_grad_norm = grad_norm_accum / max(1, grad_norm_batches)
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
            "nonfinite_grad_batches": nonfinite_batches,
            "clipped_grad_batches": clipped_batches,
            "grad_clip": self.grad_clip,
            "lr_backbone": lr_b,
            "lr_heads": lr_h,
            "samples_per_sec": samples_per_sec,
            "data_wait_sec": data_wait,
            "compute_sec": compute_time,
            "time": elapsed
        }

    def validate(self) -> Dict[str, float]:
        """Held-out evaluation; empty dict when no validation loader is configured."""
        return run_validation(
            self.model, self.val_loader, self.device, autocast, self.use_amp,
            self.wp_loss_weight, self.q_loss_weight, self.lateral_loss_weight
        )

    def _build_scheduler(self, num_epochs: int) -> LambdaLR:
        return build_warmup_cosine_scheduler(
            self.optimizer, num_epochs, len(self.train_loader),
            warmup_frac=self.warmup_frac, peak_lr=self.lr_heads
        )

    def train(self, num_epochs: int = 50, save_freq: int = 5):
        """Runs the full distillation training loop with checkpointing."""
        print(f"--> Starting World on Rails Distillation Training for {num_epochs} epochs on {self.device.upper()}...")
        self.scheduler = self._build_scheduler(num_epochs)
        best_loss = float("inf")
        # Model selection follows the held-out set when there is one. Selecting on
        # training loss makes "best" mean "most memorised" for a head with 106M
        # parameters over ~9.6k frames.
        select_on = "val_loss" if self.val_loader is not None else "total_loss"

        # The frozen backbone never changes, so it ships once. Per-epoch checkpoints
        # reference it by name and carry only the trained heads.
        self.ckpt.write_frozen_backbone_once()

        for epoch in range(1, num_epochs + 1):
            metrics = self.train_epoch(epoch)
            metrics.update(self.validate())
            is_best = metrics[select_on] < best_loss

            # Checkpointing. One CPU snapshot of the trained heads is shared by every
            # file written this epoch, and the serialization itself runs in a thread,
            # so save_sec below measures only the snapshot - the disk write overlaps
            # the next epoch.
            save_start = time.time()
            trainable = self.ckpt.trainable_state_dict_cpu()
            # dict(metrics): the writer thread pickles this while the main thread is
            # still adding save_sec to the live dict below.
            common = {"epoch": epoch, "model": trainable, "metrics": dict(metrics),
                      "partial": bool(self.ckpt.frozen_keys), "frozen_ref": "frozen_backbone.pth",
                      # Stamped so evaluation rebuilds the same geometry it was trained
                      # under. Without it, load_state_dict(strict=False) happily loads a
                      # 5-token checkpoint into a 68-token model and drives on untrained
                      # positional embeddings.
                      "config": self.model_config}
            payloads = [(dict(common), os.path.join(self.save_dir, "latest_model.pth"))]

            if is_best:
                best_loss = metrics[select_on]
                payloads.append((dict(common), os.path.join(self.save_dir, "best_model.pth")))

            # Optimizer state is only needed to resume, and for AdamW it is twice the
            # size of the weights it tracks - so it rides along with the periodic
            # snapshots rather than being rewritten every epoch.
            if epoch % save_freq == 0 or epoch == num_epochs:
                payloads[0][0]["optimizer"] = to_cpu(self.optimizer.state_dict())
                payloads.append((dict(common), os.path.join(self.save_dir, f"model_epoch_{epoch:03d}.pth")))

            self.ckpt.save_async(payloads)
            metrics["save_sec"] = time.time() - save_start

            val_str = (f"VAL Loss: {metrics['val_loss']:.4f} ADE: {metrics['val_wp_ade_m']:.3f}m "
                       f"Lat: {metrics['val_wp_lateral_error_m']:.3f}m | " if "val_loss" in metrics else "")
            print(
                f"[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"Total Loss: {metrics['total_loss']:.4f} | "
                f"Q Loss: {metrics['q_loss']:.4f} | "
                f"WP Loss: {metrics['wp_loss']:.4f} | "
                f"ADE: {metrics['wp_ade_m']:.3f}m | FDE: {metrics['wp_fde_m']:.3f}m | "
                f"Lat Err: {metrics['wp_lateral_error_m']:.3f}m | Lon Err: {metrics['wp_longitudinal_error_m']:.3f}m | "
                f"{val_str}"
                f"GradNorm: {metrics['grad_norm']:.2f} "
                f"(clipped {metrics['clipped_grad_batches']}/{metrics['num_batches']}, "
                f"skipped {metrics['nonfinite_grad_batches']}) | "
                f"Samples/s: {metrics['samples_per_sec']:.1f} | Time: {metrics['time']:.2f}s "
                f"(data {metrics['data_wait_sec']:.1f}s / compute {metrics['compute_sec']:.1f}s"
                f" / save {metrics['save_sec']:.2f}s){' ★ best' if is_best else ''}"
            )

            hw = HardwareMonitor.get_metrics()
            scalar_tags = ["total_loss", "q_loss", "wp_loss", "wp_ade_m", "wp_fde_m",
                           "wp_lateral_error_m", "wp_longitudinal_error_m",
                           "grad_norm", "nonfinite_grad_batches", "clipped_grad_batches",
                           "lr_backbone", "lr_heads", "samples_per_sec",
                           "data_wait_sec", "compute_sec", "save_sec"]
            scalar_tags += [k for k in ("val_loss", "val_wp_ade_m", "val_wp_fde_m",
                                        "val_wp_lateral_error_m", "val_wp_longitudinal_error_m")
                            if k in metrics]
            for tag in scalar_tags:
                self.logger.add_scalar(f"wor/{tag}", metrics[tag], epoch)

            # Blank rather than 0.0 when validation is off - a zero here would read as
            # a perfect held-out score in any downstream plot.
            val_cols = {k: round(metrics[k], 5) for k in (
                "val_loss", "val_wp_ade_m", "val_wp_fde_m",
                "val_wp_lateral_error_m", "val_wp_longitudinal_error_m", "val_time_sec"
            ) if k in metrics}

            self.csv_logger.log_step({
                "epoch": epoch, "num_batches": metrics["num_batches"],
                **val_cols,
                "nonfinite_grad_batches": metrics["nonfinite_grad_batches"],
                "clipped_grad_batches": metrics["clipped_grad_batches"],
                "grad_clip": metrics["grad_clip"],
                "seed": "" if self.seed is None else self.seed,
                "wall_time_s": round(time.time() - self.train_start_time, 2),
                "epoch_time_sec": round(metrics["time"], 2),
                "samples_per_sec": round(metrics["samples_per_sec"], 1),
                "data_wait_sec": round(metrics["data_wait_sec"], 2),
                "compute_sec": round(metrics["compute_sec"], 2),
                "save_sec": round(metrics["save_sec"], 2),
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

        self.ckpt.await_save()
        self.csv_logger.close()
        if os.path.exists(self.csv_logger.filepath):
            self.logger.log_artifact(self.csv_logger.filepath)
        self.logger.close()
        print(f"✓ World on Rails Training completed! Checkpoints saved to: {self.save_dir}")
