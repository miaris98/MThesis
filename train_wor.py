#!/usr/bin/env python3
"""World on Rails (WoR) Distillation Training Entry Point.

Usage:
    python train_wor.py --data_dir dataset/ --epochs 50 --batch_size 32 --backbone resnet34 --pretrained 1
"""
import argparse
import os
import torch

from src.models.world_on_rails import WorldOnRailsPolicy
from src.training.wor_trainer import WorldOnRailsTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train World on Rails (WoR) Sensorimotor Driving Policy")
    parser.add_argument("--data_dir", type=str, default="dataset/wor_trajectories", help="Path to offline CARLA dataset logs")
    parser.add_argument("--save_dir", type=str, default="checkpoints/wor_resnet34", help="Directory to save model checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50"], help="Vision backbone architecture")
    parser.add_argument("--pretrained", type=int, default=1, help="Use ImageNet pretrained weights (1=True, 0=False)")
    parser.add_argument("--freeze_backbone", type=int, default=0, help="Freeze backbone weights during training (1=True, 0=False)")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr_backbone", type=float, default=1e-4, help="Learning rate for vision backbone")
    parser.add_argument("--lr_heads", type=float, default=3e-4, help="Learning rate for Q-heads and controllers")
    default_workers = 0 if os.name == "nt" else 4
    parser.add_argument("--num_workers", type=int, default=default_workers, help="DataLoader subprocess workers")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to CARLA-pretrained backbone weights (e.g., LAV, TransFuser++, WoR, or PCLA)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--synthetic_samples", type=int, default=0, help="Generate synthetic samples if real dataset is not yet downloaded")
    parser.add_argument("--wp_loss_weight", type=float, default=1.0, help="Weight of the waypoint imitation loss")
    parser.add_argument("--q_loss_weight", type=float, default=0.0, help="Weight of the Q-value distillation loss (0 for datasets without precomputed Q-values, e.g. PDM-Lite)")
    parser.add_argument("--lateral_loss_weight", type=float, default=3.0, help="Extra weight on the lateral (y) waypoint error relative to longitudinal (x) - lateral offset is what the PID controller steers from, but is typically much smaller in magnitude than forward distance, so a flat L1 loss underfits it")
    parser.add_argument("--experiment_name", type=str, default="WoR_Offline_Training", help="MLflow experiment name")
    parser.add_argument("--use_mlflow", type=int, default=1, help="Enable MLflow tracking (1=True, 0=False)")
    parser.add_argument("--mlflow_port", type=int, default=10100, help="MLflow tracking server port")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda":
        # Every batch is a fixed 256x256 image, so cuDNN can safely autotune the
        # fastest conv kernels for that exact shape instead of using generic ones.
        torch.backends.cudnn.benchmark = True

    print("=" * 65)
    print(" 🚗 World on Rails (WoR) Distillation Training Pipeline")
    print(f" Backbone:        {args.backbone.upper()} (Pretrained: {bool(args.pretrained)})")
    if args.weights_path:
        print(f" CARLA Weights:   {args.weights_path}")
    print(f" Dataset Path:    {args.data_dir}")
    print(f" Batch Size:      {args.batch_size} | Epochs: {args.epochs}")
    print(f" Device:          {args.device.upper()}")
    print("=" * 65)

    # 1. Initialize World on Rails Policy Network
    policy = WorldOnRailsPolicy(
        backbone_name=args.backbone,
        pretrained=bool(args.pretrained),
        freeze_backbone=bool(args.freeze_backbone),
        weights_path=args.weights_path
    )

    # 2. Initialize Trainer
    trainer = WorldOnRailsTrainer(
        model=policy,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        synthetic_samples=args.synthetic_samples,
        wp_loss_weight=args.wp_loss_weight,
        q_loss_weight=args.q_loss_weight,
        lateral_loss_weight=args.lateral_loss_weight,
        experiment_name=args.experiment_name,
        use_mlflow=bool(args.use_mlflow),
        mlflow_port=args.mlflow_port
    )

    # 3. Launch Training Loop
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
