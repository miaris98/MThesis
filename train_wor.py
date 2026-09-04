#!/usr/bin/env python3
"""World on Rails (WoR) Distillation Training Entry Point.

Usage:
    python train_wor.py --data_dir dataset/ --epochs 50 --batch_size 32 --backbone resnet34 --pretrained 1
"""
import argparse
import os
import torch
import torch.nn.functional as F

from src.models.world_on_rails import WorldOnRailsPolicy
from src.training.wor_trainer import WorldOnRailsTrainer
from src.training.auto_batch_size import find_max_batch_size
from src.training.gpu_cleanup import cleanup_stale_processes


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
    # Scale with the machine instead of a flat 4: JPEG decode/resize throughput is
    # what capped epoch time regardless of batch size (see auto_batch_size discussion),
    # and more parallel workers is the cheap half of the fix. Leave a few cores free
    # for the main process, CARLA (if co-running), and OS overhead.
    default_workers = 0 if os.name == "nt" else max(4, (os.cpu_count() or 8) - 4)
    parser.add_argument("--num_workers", type=int, default=default_workers, help="DataLoader subprocess workers")
    parser.add_argument("--cache_decoded", type=int, default=1, help="Cache each decoded+resized RGB frame as a sibling .npy so repeat epochs skip JPEG decode entirely (1=True, 0=False)")
    parser.add_argument("--weights_path", type=str, default=None, help="Path to CARLA-pretrained backbone weights (e.g., LAV, TransFuser++, WoR, or PCLA)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--synthetic_samples", type=int, default=0, help="Generate synthetic samples if real dataset is not yet downloaded")
    parser.add_argument("--wp_loss_weight", type=float, default=1.0, help="Weight of the waypoint imitation loss")
    parser.add_argument("--q_loss_weight", type=float, default=0.0, help="Weight of the Q-value distillation loss (0 for datasets without precomputed Q-values, e.g. PDM-Lite)")
    parser.add_argument("--lateral_loss_weight", type=float, default=3.0, help="Extra weight on the lateral (y) waypoint error relative to longitudinal (x) - lateral offset is what the PID controller steers from, but is typically much smaller in magnitude than forward distance, so a flat L1 loss underfits it")
    parser.add_argument("--experiment_name", type=str, default="WoR_Offline_Training", help="MLflow experiment name")
    parser.add_argument("--use_mlflow", type=int, default=1, help="Enable MLflow tracking (1=True, 0=False)")
    parser.add_argument("--mlflow_port", type=int, default=10100, help="MLflow tracking server port")
    parser.add_argument("--compile_model", type=int, default=0, help="Wrap the policy in torch.compile - trades a one-off compilation on the first epoch for faster steps afterwards, so it only pays off over a long run (1=True, 0=False)")
    parser.add_argument("--kill_stale", type=int, default=1, help="On startup, terminate SUSPENDED train_wor.py processes still pinning VRAM (what Ctrl+Z leaves behind). Running instances are reported but never killed (1=True, 0=False)")
    parser.add_argument("--auto_batch_size", type=int, default=0, help="Probe the largest batch size that fits in available VRAM instead of using --batch_size directly (1=True, 0=False)")
    parser.add_argument("--vram_headroom_mb", type=float, default=2048.0, help="VRAM (MB) to leave unused when --auto_batch_size is set, so other processes sharing the GPU (e.g. an online PPO/SAC trainer) still have room")
    parser.add_argument("--auto_batch_size_max", type=int, default=512, help="Upper bound the auto batch-size search won't exceed")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda":
        # Every batch is a fixed 256x256 image, so cuDNN can safely autotune the
        # fastest conv kernels for that exact shape instead of using generic ones.
        torch.backends.cudnn.benchmark = True

    # Reclaim VRAM from a previous run left suspended by Ctrl+Z before measuring
    # what's free - otherwise the batch-size probe budgets against a GPU that a
    # dormant, abandoned process is still holding most of.
    if args.kill_stale and args.device == "cuda":
        cleanup_stale_processes("train_wor.py")

    if args.auto_batch_size:
        # Probe with a throwaway model/optimizer of the same architecture - never the
        # real one - so a few synthetic gradient steps here don't perturb the pretrained
        # weights the real run is about to load.
        def _model_factory():
            return WorldOnRailsPolicy(backbone_name=args.backbone, pretrained=bool(args.pretrained),
                                       freeze_backbone=bool(args.freeze_backbone))

        def _optimizer_factory(m):
            return torch.optim.AdamW(m.parameters(), lr=args.lr_heads, weight_decay=1e-4)

        def _batch_factory(bs):
            return {
                "rgb": torch.rand(bs, 3, 256, 256, device=args.device),
                "speed": torch.rand(bs, 1, device=args.device) * 30.0,
                "command": torch.randint(0, 6, (bs,), device=args.device),
                "target_q": torch.randn(bs, 9, device=args.device),
                "target_waypoints": torch.randn(bs, 5, 2, device=args.device) * 5.0
            }

        def _loss_fn(model, batch):
            out = model(batch["rgb"], batch["speed"], batch["command"])
            loss_q = F.mse_loss(out["selected_rail_q"], batch["target_q"])
            loss_wp_x = F.l1_loss(out["selected_waypoints"][..., 0], batch["target_waypoints"][..., 0])
            loss_wp_y = F.l1_loss(out["selected_waypoints"][..., 1], batch["target_waypoints"][..., 1])
            loss_wp = loss_wp_x + args.lateral_loss_weight * loss_wp_y
            return args.q_loss_weight * loss_q + args.wp_loss_weight * loss_wp

        args.batch_size = find_max_batch_size(
            model_factory=_model_factory, optimizer_factory=_optimizer_factory,
            batch_factory=_batch_factory, loss_fn=_loss_fn, device=args.device,
            start_batch=min(8, args.batch_size), max_batch=args.auto_batch_size_max,
            headroom_mb=args.vram_headroom_mb
        )

    print("=" * 65)
    print(" 🚗 World on Rails (WoR) Distillation Training Pipeline")
    print(f" Backbone:        {args.backbone.upper()} (Pretrained: {bool(args.pretrained)})")
    if args.weights_path:
        print(f" CARLA Weights:   {args.weights_path}")
    print(f" Dataset Path:    {args.data_dir}")
    print(f" Batch Size:      {args.batch_size}{' (auto)' if args.auto_batch_size else ''} | Epochs: {args.epochs}")
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
        cache_decoded=bool(args.cache_decoded),
        compile_model=bool(args.compile_model),
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
