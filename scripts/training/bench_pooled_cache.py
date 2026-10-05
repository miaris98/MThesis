"""TODO A59 verification on real data: (1) parity - the policy's outputs on the same real frames with the live encoder vs the pooled cache; (2) speed - seconds per
training step (forward + backward of the head, bf16, batch 256) with JPEG decode + overlay + encoder (live) vs the cache.

    python scripts/training/bench_pooled_cache.py --checkpoint /workspace/checkpoints/carla_armJ_ft_obst/model_epoch_020.pth --data_dir <dir> --cache <prefix> \
        --img_size 288x768 --route_key route_original --use_augmented_camera 1 --steps 30 --device cuda:1
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--img_size", default="288x768")
    p.add_argument("--route_key", default="route_original")
    p.add_argument("--use_augmented_camera", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true", help="fp32 everywhere: separates the cache's own error from bf16 autocast noise")
    a = p.parse_args()
    os.environ["WOR_ROUTE_KEY"] = a.route_key
    from src.models.world_on_rails.wor_loader import load_wor_model
    from src.training.wor_dataset import WorldOnRailsDataset
    h, w = (int(x) for x in a.img_size.split("x"))
    dev = torch.device(a.device)
    arch = "qwen30m"
    model = load_wor_model(checkpoint_path=a.checkpoint, policy_arch=arch, backbone_name="regnety_032", device=dev) if "device" in load_wor_model.__code__.co_varnames else \
        load_wor_model(checkpoint_path=a.checkpoint, policy_arch=arch, backbone_name="regnety_032").to(dev)
    model.train()
    kw = dict(data_dir=a.data_dir, img_size=(h, w), crop_bottom_frac=0.0, route_overlay=True, route_points=4, use_augmented_camera=bool(a.use_augmented_camera), cache_decoded=False)
    os.environ.pop("WOR_POOLED_CACHE", None)
    live = WorldOnRailsDataset(is_train=True, **kw)
    os.environ["WOR_POOLED_CACHE"] = a.cache
    pooled = WorldOnRailsDataset(is_train=True, **kw)
    assert len(live) == len(pooled)
    amp = torch.bfloat16 if (dev.type == "cuda" and not a.no_amp and torch.cuda.is_bf16_supported(including_emulation=False)) else torch.float32

    def fwd(batch, use_feats):
        sp, cmd, rt = (batch[k].to(dev) for k in ("speed", "command", "route"))
        rgb = None if use_feats else batch["rgb"].to(dev).permute(0, 3, 1, 2).float().div_(255.0)
        vf = batch["vision_features"].to(dev) if use_feats else None
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp != torch.float32):
            return model(rgb, sp, cmd, rt, vision_features=vf)

    # 1. parity on 128 frames (eval mode: no dropout noise)
    idx = list(range(0, len(live), max(1, len(live) // 128)))[:128]
    from torch.utils.data import Subset
    bl = next(iter(DataLoader(Subset(live, idx), batch_size=len(idx), num_workers=8)))
    bp = next(iter(DataLoader(Subset(pooled, idx), batch_size=len(idx), num_workers=8)))
    model.eval()
    with torch.no_grad():
        ol, op = fwd(bl, False), fwd(bp, True)
    d_wp = (ol["selected_waypoints"].float() - op["selected_waypoints"].float()).abs()
    d_ts = (ol["target_speed_logits"].float() - op["target_speed_logits"].float()).abs() if "target_speed_logits" in ol else torch.zeros(1)
    print(f"PARITY on {len(idx)} real frames: waypoints max|diff| {d_wp.max():.4f} m (mean {d_wp.mean():.5f}, scale {ol['selected_waypoints'].float().abs().mean():.2f} m); "
          f"speed logits max|diff| {d_ts.max():.4f}", flush=True)
    # noise floor: the same live forward at another batch split (bf16 kernels are not bitwise reproducible across shapes)
    with torch.no_grad():
        half = len(idx) // 2
        parts = [fwd({k: v[i:i + half] for k, v in bl.items()}, False)["selected_waypoints"].float() for i in (0, half)]
    d_nf = (ol["selected_waypoints"].float() - torch.cat(parts)).abs()
    print(f"NOISE FLOOR (live vs live at batch {half}): waypoints max|diff| {d_nf.max():.4f} m (mean {d_nf.mean():.5f})", flush=True)
    model.train()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)

    # 2. speed
    for name, ds, use_feats in (("live encoder", live, False), ("pooled cache", pooled, True)):
        ld = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=False)
        it = iter(ld)
        t_data = t_step = 0.0
        for i in range(a.steps + 3):
            t0 = time.time()
            batch = next(it)
            t1 = time.time()
            out = fwd(batch, use_feats)
            loss = out["selected_waypoints"].float().abs().mean() + (out["target_speed_logits"].float().logsumexp(-1).mean() if "target_speed_logits" in out else 0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()
            if i >= 3:  # skip warm-up
                t_data += t1 - t0
                t_step += t2 - t1
        n = a.steps
        print(f"SPEED {name:13s}: {(t_data + t_step) / n:.3f} s/step  (wait for data {t_data / n:.3f} s, GPU step {t_step / n:.3f} s) -> {len(ds) / a.batch_size * (t_data + t_step) / n / 60:.1f} min per {len(ds)}-frame epoch", flush=True)


if __name__ == "__main__":
    main()
