"""World on Rails (WoR) Distillation Trainer.

Trains the sensorimotor vision policy to predict optimal Q-values and waypoints
using differential learning rates, PyTorch AMP (Automatic Mixed Precision),
and telemetry tracking.
"""
from typing import Dict, Optional, Tuple
import os
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.world_on_rails.wor_policy import WorldOnRailsPolicy
from src.training.wor_dataset import create_wor_dataloader


try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    from torch.amp import GradScaler, autocast


class WorldOnRailsTrainer:
    """
    Trainer for World on Rails Policy Distillation.
    """
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
        synthetic_samples: int = 0
    ):
        self.model = model.to(device)
        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.save_dir = save_dir
        self.device = device
        self.use_amp = use_amp and (device == "cuda")
        self.wp_loss_weight = wp_loss_weight
        # Datasets without precomputed Q-values (e.g. PDM-Lite) leave target_q at
        # zero, so q_loss_weight defaults to 0 to avoid supervising toward zero.
        self.q_loss_weight = q_loss_weight

        os.makedirs(save_dir, exist_ok=True)
        self.telemetry_csv = os.path.join(save_dir, "wor_training_telemetry.csv")
        self._init_csv()

        # 1. DataLoaders
        self.train_loader = create_wor_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=True,
            synthetic_samples=synthetic_samples
        )

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

    def _init_csv(self):
        """Initializes CSV telemetry header."""
        if not os.path.exists(self.telemetry_csv):
            with open(self.telemetry_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "step", "total_loss", "q_loss", "wp_loss",
                    "lr_backbone", "lr_heads", "time_sec"
                ])

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full training epoch."""
        self.model.train()
        total_loss_accum = 0.0
        q_loss_accum = 0.0
        wp_loss_accum = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            rgb = batch["rgb"].to(self.device)
            speed = batch["speed"].to(self.device)
            command = batch["command"].to(self.device)
            target_q = batch["target_q"].to(self.device)
            target_wp = batch["target_waypoints"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                out = self.model(rgb, speed, command)
                
                # 1. Q-value distillation loss (MSE on selected rail Q-values)
                pred_q = out["selected_rail_q"]
                loss_q = F.mse_loss(pred_q, target_q)

                # 2. Waypoint imitation loss
                pred_wp = out["selected_waypoints"]
                loss_wp = F.l1_loss(pred_wp, target_wp)

                total_loss = self.q_loss_weight * loss_q + self.wp_loss_weight * loss_wp

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            total_loss_accum += total_loss.item()
            q_loss_accum += loss_q.item()
            wp_loss_accum += loss_wp.item()
            num_batches += 1

        avg_loss = total_loss_accum / max(1, num_batches)
        avg_q_loss = q_loss_accum / max(1, num_batches)
        avg_wp_loss = wp_loss_accum / max(1, num_batches)
        elapsed = time.time() - start_time

        lr_b = self.optimizer.param_groups[0]["lr"]
        lr_h = self.optimizer.param_groups[1]["lr"]

        # Log telemetry to CSV
        with open(self.telemetry_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, num_batches, f"{avg_loss:.5f}", f"{avg_q_loss:.5f}",
                f"{avg_wp_loss:.5f}", f"{lr_b:.2e}", f"{lr_h:.2e}", f"{elapsed:.2f}"
            ])

        return {
            "epoch": epoch,
            "total_loss": avg_loss,
            "q_loss": avg_q_loss,
            "wp_loss": avg_wp_loss,
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

            print(
                f"[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"Total Loss: {metrics['total_loss']:.4f} | "
                f"Q Loss: {metrics['q_loss']:.4f} | "
                f"WP Loss: {metrics['wp_loss']:.4f} | "
                f"Time: {metrics['time']:.2f}s"
            )

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

        print(f"✓ World on Rails Training completed! Checkpoints saved to: {self.save_dir}")
