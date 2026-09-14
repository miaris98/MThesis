#!/usr/bin/env python3
"""World on Rails (WoR) Distillation Training Entry Point.

Usage:
    python train_wor.py --data_dir dataset/ --epochs 50 --batch_size 32 --backbone resnet34 --pretrained 1
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F

from src.config import paths
from src.models.world_on_rails import WorldOnRailsPolicy, QwenWorldOnRailsPolicy
from src.training.wor_trainer import WorldOnRailsTrainer
from src.training.auto_batch_size import find_max_batch_size
from src.training.gpu_cleanup import cleanup_stale_processes
from src.training.seeding import seed_everything


def _parse_img_size(spec: str):
    """'192x512' -> (192, 512). Height first, matching the (H, W) convention everywhere else.

    Worth being explicit about because the two numbers are not interchangeable here: the source
    render is 1024x512 (W x H), so a spec whose aspect ratio does not match the post-crop source
    re-introduces the geometric distortion this flag exists to remove.
    """
    try:
        h, w = spec.lower().split("x")
        return int(h), int(w)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"--img_size must look like HxW (e.g. 192x512), got {spec!r}") from exc


def _parse_vision_grid(spec):
    """'8' -> (8, 8); '6x16' -> (6, 16); '0' -> (0, 0) for the globally-pooled ablation."""
    if isinstance(spec, (tuple, list)):
        return int(spec[0]), int(spec[1])
    s = str(spec).lower()
    if "x" in s:
        h, w = s.split("x")
        return int(h), int(w)
    v = int(s)
    return v, v


def parse_args():
    parser = argparse.ArgumentParser(description="Train World on Rails (WoR) Sensorimotor Driving Policy")
    parser.add_argument("--data_dir", type=str, default="dataset/wor_trajectories", help="Path to offline CARLA dataset logs")
    parser.add_argument("--save_dir", type=str, default=str(paths.checkpoints_dir() / "wor_resnet34"),
                        help="Directory to save model checkpoints, TensorBoard events and telemetry. Defaults under the machine's experiment root (external disk locally, /workspace on vast.ai) - see src/config/paths.py")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50", "regnety_032"], help="Vision backbone architecture. 'regnety_032' selects the CARLA-pretrained TransFuser++ encoder (src/models/world_on_rails/carla_encoder.py) and requires --weights_path; the resnet options are the ImageNet-pretrained torchvision stacks")
    parser.add_argument("--img_size", type=str, default="256x256", help="Input resolution as HxW. The PDM-Lite source render is 1024x512, so the 256x256 default squashes horizontal geometry ~2x relative to vertical and throws away 4x the horizontal resolution; '192x512' with --crop_bottom_frac 0.25 reproduces TransFuser++'s aspect-preserving 1024x384 crop at half scale")
    parser.add_argument("--crop_bottom_frac", type=float, default=0.0, help="Fraction of image height cut from the bottom before resize. 0.25 removes the ego bonnet from PDM-Lite's 1024x512 render, matching the 1024x384 crop the CARLA-pretrained encoder was trained on. Part of the decode-cache key, so it cannot silently serve a differently-cropped array")
    parser.add_argument("--max_batches", type=int, default=0, help="Stop each epoch after this many batches (0 = full epoch). For smoke tests that measure per-batch cost and extrapolate a full run without paying for one")
    parser.add_argument("--route_overlay", type=int, default=0, help="Draw the planned route into the camera image like a reversing camera's guide lines, so route and road share a spatial frame before the frozen encoder sees either (1=True, 0=False). Uses the same subsampled route the route MLP gets, so it adds no information the policy did not already have - only a spatial presentation of it. Must match at evaluation; the agent reads it from run_config.json")
    parser.add_argument("--target_speed_loss_weight", type=float, default=0.0, help="Weight of the auxiliary 8-bin target-speed classification loss (0 disables the head entirely). Gives the policy an explicit longitudinal output including an exactly-zero class, instead of leaving speed to be inferred from waypoint spacing - which can express 'slow' but only approaches 'stopped', and cannot express 'stationary while steering' at all")
    parser.add_argument("--policy_arch", type=str, default="cnn", choices=["cnn", "qwen10m", "qwen30m", "qwen100m", "qwen500m", "qwen900m"], help="Decision-head architecture on top of the frozen vision encoder: 'cnn' is the original WoR SpatialQHead (conv+MLP); 'qwen*' swaps it for a Qwen-style self-attention transformer trunk (see qwen_wor_policy.py), sized 10M/30M/100M/500M/900M params (the small sizes matter here - the offline dataset is ~9,600 frames, and the 100M trunk underfits it), still predicting waypoints for the same PIDController")
    parser.add_argument("--pretrained", type=int, default=1, help="Use ImageNet pretrained weights (1=True, 0=False)")
    parser.add_argument("--freeze_backbone", type=int, default=1, help="Freeze the pretrained vision backbone so only the policy heads train (1=True, 0=False). On by default: training the vision model is out of scope here, and fine-tuning it is also the bulk of the compute cost")
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
    parser.add_argument("--heading_loss_weight", type=float, default=0.0, help="Weight on the segment-direction (1 - cosine) error between the predicted path and the expert's. The waypoint L1 above is dominated by longitudinal displacement - on the four-town runs it is 0.433 of a 0.589 loss, i.e. 73%% - which the PID's steering never reads, and which is close to the integral of the speed scalar the policy is already given. This term is scale-free, so it cannot be swamped by that magnitude, and it supervises the quantity the controller actually aims at. 0 reproduces the objective the existing results were measured with")
    parser.add_argument("--curvature_loss_weight", type=float, default=0.0, help="Weight on the L1 between the second differences of the predicted and expert paths. Per-waypoint L1 is indifferent to point-to-point zig-zag that integrates to the same positions; the PID's derivative term is not, and reacts to it as steering chatter. 0 reproduces the previous objective")
    parser.add_argument("--experiment_name", type=str, default="WoR_Offline_Training", help="MLflow experiment name")
    parser.add_argument("--use_mlflow", type=int, default=1, help="Enable MLflow tracking (1=True, 0=False)")
    parser.add_argument("--mlflow_port", type=int, default=10100, help="MLflow tracking server port")
    parser.add_argument("--route_points", type=int, default=4, help="How many ego-frame route points to condition the policy on. PDM-Lite's command enum is LANEFOLLOW on every frame, so this is the policy's only navigation input - without it the model cannot tell a left turn from a right one and learns to drive straight. Higher values leak more of the answer (the full 20-point route correlates ~0.84 with the lateral target); use this to ablate that")
    parser.add_argument("--vision_grid", type=str, default="8", help="Qwen heads only. Accepts a single int for a square grid or 'HxW' (e.g. '6x16') to keep the encoder's native rectangular layout - required once --img_size is not square, since pooling a 6x16 feature map to 8x8 re-imposes the very horizontal squash the aspect-correct crop removes. Side length of the vision token grid handed to the transformer: 8 forwards all 64 cells of the encoder's 8x8 feature map as separate tokens (68-token sequence), 4 pools to 4x4 first (20 tokens), and 0 restores the original single globally-averaged vision token (5 tokens). 0 is the configuration the first Qwen runs used and is spatially blind by construction - AdaptiveAvgPool2d((1,1)) gives every cell an identical gradient - so it exists only as an ablation. Costs wall time: the trunk's compute scales with sequence length, so 8 is roughly an order of magnitude slower per epoch than 0")
    parser.add_argument("--pool_vision", type=int, default=0, help="CNN head only. Collapses the encoder feature map to 1x1 before SpatialQHead, giving the CNN the same spatially-blind input as --vision_grid 0. The matching ablation on the CNN side: without it, a CNN-beats-transformer result confounds architecture family with the fact that only one head was shown where things are in the frame (1=True, 0=False)")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Global gradient-norm clip threshold. The 5.0 default was set against the CNN head, whose epoch-mean gradient norm drops below it around epoch 17; the 106M Qwen trunk stayed above it on all 50 epochs of its first run, so the two were not being given comparable effective step sizes")
    parser.add_argument("--warmup_frac", type=float, default=0.05, help="Fraction of total optimizer steps spent linearly warming the learning rate up before cosine decay begins. 0 reproduces the previous schedule (full LR from batch 1), which a conv head tolerates and a 12-layer pre-norm transformer generally does not")
    parser.add_argument("--decay_gates_and_norms", type=int, default=0, help="Apply AdamW weight decay to gates, norm gains, embeddings and learned tokens as well as weight matrices (1=True, 0=False). The old behaviour, kept for reproducibility. It decays the transformer's 24 alpha residual gates toward zero, i.e. penalizes the model for letting each block contribute at all")
    parser.add_argument("--val_data_dir", type=str, default=None, help="Separate held-out dataset directory. Preferred over --val_split, which divides frames rather than routes")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of --data_dir held out for validation when --val_data_dir is not given, split on route boundaries where the layout exposes them. 0 disables validation entirely, which also means 'best' is selected on training loss. 0.15 rather than 0.05 because the held-out number is what an ablation is decided on: 5%% of ~9,600 frames is ~480, and split on route boundaries that can be one or two routes, which is far too noisy to separate two components")
    parser.add_argument("--split_seed", type=int, default=0, help="Seed for the deterministic train/validation split")
    parser.add_argument("--num_folds", type=int, default=1,
                        help="K-fold cross-validation over routes. With K>1 the shuffled route list is "
                             "cut into K disjoint blocks and --fold selects which is held out, so K runs "
                             "score the model on every route exactly once instead of on one 15%% draw. "
                             "Overrides --val_split.")
    parser.add_argument("--fold", type=int, default=0, help="Which fold to hold out (0..num_folds-1)")
    parser.add_argument("--seed", type=int, default=0, help="Master seed for weight initialization, dropout and data order. Nothing in this pipeline was seeded before, which is fine for one run and fatal for a comparison: an ablation cannot attribute a difference to a component until the spread between two identical runs is known. Vary this to measure that spread")
    parser.add_argument("--deterministic", type=int, default=0, help="Additionally pin cuDNN kernel selection and refuse nondeterministic CUDA kernels, for bitwise reproducibility (1=True, 0=False). Costs throughput - it disables the cuDNN autotuner - so use it to reproduce one run, not to run a sweep")
    parser.add_argument("--run_label", type=str, default=None, help="Human-readable name for this run, recorded in run_config.json and used by compare_wor_runs.py to group repeats of the same configuration. Defaults to the save_dir basename")
    parser.add_argument("--target_speed_input", type=str, default="policy",
                        choices=["policy", "state"],
                        help="What the target-speed head reads. policy = the trunk post-attention "
                             "representation, which has seen the image. state = the legacy "
                             "speed+route+command vector, which has not, so it cannot see a red "
                             "light or a lead vehicle. Kept only to reproduce earlier runs.")
    parser.add_argument("--use_rail_q", type=int, default=0,
                        help="Build the rail-Q / Q-map heads. They only train when --q_loss_weight "
                             "> 0 and the dataset carries real q_values; PDM-Lite target_q is all "
                             "zeros, so at the defaults they get no gradient, are never read by "
                             "act(), and the reported Q Loss means nothing (1=True, 0=False).")
    parser.add_argument("--ray_geometry", type=int, default=0,
                        help="Append per-cell camera geometry (ray bearing/elevation and the "
                             "flat-ground intersection) to the vision features, derived from "
                             "src/config/camera.py. No new data, one cached tensor, and identical "
                             "for both arms (1=True, 0=False).")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Run held-out validation every Nth epoch instead of every epoch (TF++ uses 5). "
                             "Saves ~10%% of epoch wall time, at the cost of coarsening which checkpoint "
                             "best_model.pth can select - best is only chosen among validated epochs. "
                             "The final epoch is always validated.")
    parser.add_argument("--feature_cache_tag", type=str, default=None,
                        help="Read the frozen backbone's output from the sidecar cache built by "
                             "build_feature_cache.py instead of running the encoder every step. Pass the "
                             "backbone name (e.g. regnety_032); it must match the --backbone/--img_size/"
                             "--crop_bottom_frac/--route_overlay the cache was built with, and every frame "
                             "must already be cached - a miss is a hard error, never a silent fallback.")
    parser.add_argument("--compile_model", type=int, default=0, help="Wrap the policy in torch.compile - trades a one-off compilation on the first epoch for faster steps afterwards, so it only pays off over a long run (1=True, 0=False)")
    parser.add_argument("--kill_stale", type=int, default=1, help="On startup, terminate SUSPENDED train_wor.py processes still pinning VRAM (what Ctrl+Z leaves behind). Running instances are reported but never killed (1=True, 0=False)")
    parser.add_argument("--auto_batch_size", type=int, default=0, help="Probe the largest batch size that fits in available VRAM instead of using --batch_size directly (1=True, 0=False)")
    parser.add_argument("--vram_headroom_mb", type=float, default=2048.0, help="VRAM (MB) to leave unused when --auto_batch_size is set, so other processes sharing the GPU (e.g. an online PPO/SAC trainer) still have room")
    parser.add_argument("--auto_batch_size_max", type=int, default=512, help="Upper bound the auto batch-size search won't exceed")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a model_epoch_*.pth written by a prior run of this exact config - loads the trainable-head weights and optimizer state and continues at checkpoint['epoch']+1 with a fast-forwarded LR schedule, instead of starting over at epoch 1. Use a dated model_epoch_*.pth, not latest_model.pth - only the dated snapshots reliably carry optimizer state (see WorldOnRailsTrainer.train's docstring)")
    parser.add_argument("--save_freq", type=int, default=5, help="Epoch interval between dated model_epoch_*.pth snapshots (the ones --resume_from targets) and optimizer-state saves. Lower values bound how much a stopped/interrupted run can lose, at the cost of writing (and briefly holding in memory) the optimizer state - roughly 2x the trainable weights' size for AdamW - more often")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolved once, here, because three separate things downstream need the real input shape:
    # the auto-batch-size probe (which must allocate the same shape the run will), the banner's
    # feature-grid line, and the trainer itself. It used to be parsed only at the trainer call
    # and assumed to be 256x256 everywhere else, which was true until --img_size existed.
    img_h, img_w = _parse_img_size(args.img_size)

    # --use_rail_q now defaults off, so asking for a Q loss without the heads that produce it
    # would silently train nothing: the term would be absent from the output and contribute a
    # flat zero. Refuse rather than let a run report a q_loss that means nothing - which is
    # exactly what the old always-built heads did.
    if args.q_loss_weight > 0 and not args.use_rail_q:
        raise SystemExit(
            f"--q_loss_weight {args.q_loss_weight} needs --use_rail_q 1: the rail-Q heads are "
            "not built by default. Note PDM-Lite carries no q_values, so target_q is all "
            "zeros and this loss has nothing to learn from on that dataset.")

    if args.device == "cuda":
        # Every batch is a fixed {img_h}x{img_w} image, so cuDNN can safely autotune the
        # fastest conv kernels for that exact shape instead of using generic ones.
        # seed_everything(deterministic=True) turns this back off, which is the
        # trade it is documented to make.
        torch.backends.cudnn.benchmark = True

    # Seeded before any model is constructed, so weight init is part of what the seed
    # controls - the auto-batch-size probe below builds throwaway models too.
    seed_everything(args.seed, deterministic=bool(args.deterministic))

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
            if args.policy_arch == "cnn":
                return WorldOnRailsPolicy(backbone_name=args.backbone, pretrained=bool(args.pretrained),
                                           freeze_backbone=bool(args.freeze_backbone),
                                           route_points=args.route_points,
                                           pool_vision=bool(args.pool_vision))
            return QwenWorldOnRailsPolicy(backbone_name=args.backbone, pretrained=bool(args.pretrained),
                                           freeze_backbone=bool(args.freeze_backbone),
                                           route_points=args.route_points,
                                           model_size=args.policy_arch.replace("qwen", ""),
                                           vision_grid=_parse_vision_grid(args.vision_grid))

        def _optimizer_factory(m):
            return torch.optim.AdamW(m.parameters(), lr=args.lr_heads, weight_decay=1e-4)

        def _batch_factory(bs):
            return {
                "rgb": torch.rand(bs, 3, img_h, img_w, device=args.device),
                "speed": torch.rand(bs, 1, device=args.device) * 30.0,
                "command": torch.randint(0, 6, (bs,), device=args.device),
                "route": torch.randn(bs, args.route_points, 2, device=args.device),
                "target_q": torch.randn(bs, 9, device=args.device),
                "target_waypoints": torch.randn(bs, 5, 2, device=args.device) * 5.0
            }

        def _loss_fn(model, batch):
            out = model(batch["rgb"], batch["speed"], batch["command"], batch["route"])
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
    print(f" Policy Head:     {args.policy_arch.upper()}")
    if args.policy_arch == "cnn":
        # Derived, not hardcoded: every backbone here reduces by 32, so the map the CNN head
        # actually receives is img_size/32. The literal '8x8' this used to print was only ever
        # true for a 256x256 input, so at 192x512 a run directory would have described itself
        # with a feature map it never saw (6x16).
        print(f" Vision input:    {'1x1 GLOBALLY POOLED (spatially blind ablation)' if args.pool_vision else f'{img_h // 32}x{img_w // 32} spatial feature map'}")
    else:
        _gh, _gw = _parse_vision_grid(args.vision_grid)
        n_tok = 1 if min(_gh, _gw) <= 0 else _gh * _gw
        print(f" Vision input:    {_gh}x{_gw} -> {n_tok} token(s) + 4 state = "
              f"{n_tok + 4}-token sequence"
              f"{'  [GLOBALLY POOLED - spatially blind ablation]' if min(_gh, _gw) <= 0 else ''}")
    print(f" Optimizer:       clip {args.grad_clip} | warmup {args.warmup_frac:.0%} of steps | "
          f"decay on gates/norms: {bool(args.decay_gates_and_norms)}")
    print(f" Seed:            {args.seed}{' (deterministic)' if args.deterministic else ''}")
    print(f" Validation:      {args.val_data_dir or (f'{args.val_split:.0%} split of --data_dir' if args.val_split > 0 else 'NONE (best selected on train loss)')}")
    print(f" Backbone:        {args.backbone.upper()} (Pretrained: {bool(args.pretrained)})")
    print(f" Training:        {'policy heads only - vision backbone FROZEN' if args.freeze_backbone else 'policy heads + vision backbone (fine-tuning vision!)'}")
    if args.weights_path:
        print(f" CARLA Weights:   {args.weights_path}")
    print(f" Dataset Path:    {args.data_dir}")
    print(f" Batch Size:      {args.batch_size}{' (auto)' if args.auto_batch_size else ''} | Epochs: {args.epochs}")
    print(f" Device:          {args.device.upper()}")
    print("=" * 65)

    # 1. Initialize World on Rails Policy Network
    if args.policy_arch == "cnn":
        policy = WorldOnRailsPolicy(
            backbone_name=args.backbone,
            pretrained=bool(args.pretrained),
            freeze_backbone=bool(args.freeze_backbone),
            weights_path=args.weights_path,
            route_points=args.route_points,
            pool_vision=bool(args.pool_vision),
            use_target_speed=args.target_speed_loss_weight > 0,
            target_speed_input=args.target_speed_input,
            use_rail_q=bool(args.use_rail_q),
            use_ray_geometry=bool(args.ray_geometry),
            crop_bottom_frac=args.crop_bottom_frac
        )
    else:
        policy = QwenWorldOnRailsPolicy(
            backbone_name=args.backbone,
            pretrained=bool(args.pretrained),
            freeze_backbone=bool(args.freeze_backbone),
            weights_path=args.weights_path,
            route_points=args.route_points,
            model_size=args.policy_arch.replace("qwen", ""),
            vision_grid=_parse_vision_grid(args.vision_grid),
            use_target_speed=args.target_speed_loss_weight > 0,
            target_speed_input=args.target_speed_input,
            use_rail_q=bool(args.use_rail_q),
            use_ray_geometry=bool(args.ray_geometry),
            crop_bottom_frac=args.crop_bottom_frac
        )

    # 2. Initialize Trainer
    trainer = WorldOnRailsTrainer(
        model=policy,
        data_dir=args.data_dir,
        val_data_dir=args.val_data_dir,
        val_split=args.val_split,
        split_seed=args.split_seed,
        fold=args.fold,
        num_folds=args.num_folds,
        seed=args.seed,
        grad_clip=args.grad_clip,
        warmup_frac=args.warmup_frac,
        decay_gates_and_norms=bool(args.decay_gates_and_norms),
        save_dir=args.save_dir,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        synthetic_samples=args.synthetic_samples,
        cache_decoded=bool(args.cache_decoded),
        compile_model=bool(args.compile_model),
        feature_cache_tag=args.feature_cache_tag,
        val_every=args.val_every,
        route_points=args.route_points,
        img_size=(img_h, img_w),
        crop_bottom_frac=args.crop_bottom_frac,
        route_overlay=bool(args.route_overlay),
        max_batches=args.max_batches,
        target_speed_loss_weight=args.target_speed_loss_weight,
        wp_loss_weight=args.wp_loss_weight,
        q_loss_weight=args.q_loss_weight,
        lateral_loss_weight=args.lateral_loss_weight,
        heading_loss_weight=args.heading_loss_weight,
        curvature_loss_weight=args.curvature_loss_weight,
        experiment_name=args.experiment_name,
        use_mlflow=bool(args.use_mlflow),
        mlflow_port=args.mlflow_port
    )

    # 3. Record the full configuration next to the telemetry, so a run directory is
    # self-describing without MLflow having been reachable. compare_wor_runs.py reads
    # this to group and label runs; a sweep whose rows cannot be traced back to the
    # flags that produced them is not an experiment.
    os.makedirs(args.save_dir, exist_ok=True)
    run_config = dict(vars(args))
    run_config["run_label"] = args.run_label or os.path.basename(os.path.normpath(args.save_dir))
    run_config["batch_size_effective"] = args.batch_size
    with open(os.path.join(args.save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, sort_keys=True, default=str)

    # 4. Launch Training Loop
    trainer.train(num_epochs=args.epochs, save_freq=args.save_freq, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
