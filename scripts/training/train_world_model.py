"""TODO A73 (2026-11-01): train the latent world model offline on the PDM-Lite logs (src/training/latent_world_model_train.py).

Inputs: a pooled feature cache of the frozen regnety encoder (scripts/training/build_pooled_cache.py, the A59 cache: --pooled_cache prefix list)
and the A71 label files built with the A72 transition fields (--q_labels TAG). Validation holds out whole routes (--val_frac). Each epoch writes
wm_epoch_NNN.pth with the model, its EMA target, both configs and the held-out routes, which is everything scripts/analysis/a73_world_model_gate.py
needs. MLflow experiment "A73_world_model".

    python scripts/training/train_world_model.py --data_dir /workspace/dataset/wor_trajectories --pooled_cache /workspace/pooled/wor_... \\
        --q_labels qcf_v4 --trunk qwen --size 30m --save_dir /workspace/checkpoints/A73_q
"""
import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.models.world_on_rails.latent_world_model import LatentWorldModel, WorldModelConfig  # noqa: E402
from src.training.latent_world_model_train import (LatentWindowDataset, WMLossConfig, policy_improvement_loss, unroll_losses,  # noqa: E402
                                                   update_target)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pooled_cache", required=True, help="comma-separated pooled-cache prefixes")
    ap.add_argument("--q_labels", required=True, help="label tag with the A72 transition fields (a_expert, r_expert)")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--trunk", default="qwen", choices=["qwen", "conv", "mlp"], help="qwen = A73-Q, conv = A73-W")
    ap.add_argument("--size", default=None, help="qwen_wor_policy._MODEL_SIZES key (30m = the CARLA/Atari network); overrides the dims")
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--repr_depth", type=int, default=2)
    ap.add_argument("--dyn_depth", type=int, default=8)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--ffn_dim", type=int, default=1536)
    ap.add_argument("--n_ensemble", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--penalty", type=float, default=1.0, help="MOPO lambda on the ensemble spread")
    ap.add_argument("--bc_weight", type=float, default=0.2)
    ap.add_argument("--imag_weight", type=float, default=1.0)
    ap.add_argument("--imag_horizon", type=int, default=5, help="imagined steps behind the policy target (1 = one-step lookahead)")
    ap.add_argument("--pi_batch", type=int, default=0, help="states per batch with the lookahead target (0 = all; BC uses every state)")
    ap.add_argument("--tf32", type=int, default=0, help="1 = TF32 matmuls (1.5x on the RTX 5000 Ada box, 2026-11-02)")
    ap.add_argument("--lr", type=float, default=1e-4, help="AdamW; B66/B67: <= 1e-4 for transformer RL heads")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--target_tau", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--use_mlflow", type=int, default=1)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(a.tf32)
    os.makedirs(a.save_dir, exist_ok=True)
    prefixes = [p for p in a.pooled_cache.split(",") if p]
    full = LatentWindowDataset(a.data_dir, prefixes, a.q_labels, K=a.K)
    routes = sorted({os.path.dirname(os.path.dirname(w[0])) for w in full.windows})
    random.Random(a.seed).shuffle(routes)
    val_routes = sorted(routes[:int(round(a.val_frac * len(routes)))])
    train_ds = LatentWindowDataset(a.data_dir, prefixes, a.q_labels, K=a.K, routes=set(routes) - set(val_routes))
    val_ds = LatentWindowDataset(a.data_dir, prefixes, a.q_labels, K=a.K, routes=set(val_routes)) if val_routes else None
    print(f"--> {len(train_ds)} train / {len(val_ds) if val_ds else 0} val windows, {len(routes)} routes ({len(val_routes)} held out)", flush=True)
    kw = dict(batch_size=a.batch_size, num_workers=a.num_workers, pin_memory=a.device.startswith("cuda"),
              persistent_workers=a.num_workers > 0, prefetch_factor=2 if a.num_workers > 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) >= a.batch_size, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw) if val_ds else None

    m0 = full.meta[0]
    mcfg = WorldModelConfig(feat_channels=m0["channels"], grid=tuple(m0["grid"]), trunk=a.trunk, size=a.size, embed_dim=a.embed_dim,
                            repr_depth=a.repr_depth, dyn_depth=a.dyn_depth, num_heads=a.num_heads, ffn_dim=a.ffn_dim, n_ensemble=a.n_ensemble)
    model = LatentWorldModel(mcfg).to(a.device)
    target = LatentWorldModel(mcfg).to(a.device)
    target.load_state_dict(model.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    lcfg = WMLossConfig(gamma=a.gamma, penalty=a.penalty, bc_weight=a.bc_weight, imag_weight=a.imag_weight, horizon=a.imag_horizon,
                        pi_batch=a.pi_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total_steps = max(1, a.epochs * len(train_loader))
    warm = max(1, int(a.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    run_config = {**vars(a), "model_config": dataclasses.asdict(mcfg.resolved()), "loss_config": dataclasses.asdict(lcfg)}
    json.dump(run_config, open(os.path.join(a.save_dir, "run_config.json"), "w"), indent=1)

    ml = None
    if a.use_mlflow:
        try:
            import mlflow
            from src.config.paths import mlflow_tracking_uri
            mlflow.set_tracking_uri(mlflow_tracking_uri())
            mlflow.set_experiment("A73_world_model")
            mlflow.start_run(run_name=os.path.basename(os.path.normpath(a.save_dir)))
            mlflow.log_params({k: str(v) for k, v in vars(a).items()})
            ml = mlflow
        except Exception as e:
            print("mlflow skipped:", e)

    def run_epoch(loader, train: bool):
        model.train(train)
        sums, n = {}, 0
        for b in loader:
            f, acts, rew, val = (b[k].to(a.device, non_blocking=True) for k in ("feats", "actions", "rewards", "valid"))
            with torch.set_grad_enabled(train):
                l = unroll_losses(model, target, f, acts, rew, val, lcfg)
                p = policy_improvement_loss(model, model.encode(f[:, 0]), acts[:, 0], lcfg)
                total = l["total"] + p["total"]
            if train:
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step()
                sched.step()
                update_target(target, model, a.target_tau)
            for k, v in {**l, **{f"{k}": v for k, v in p.items() if k != "total"}, "total": total}.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(train_loader, True)
        metrics = {f"train_{k}": v for k, v in tr.items()}
        if val_loader is not None:
            metrics.update({f"val_{k}": v for k, v in run_epoch(val_loader, False).items()})
        metrics["epoch_sec"] = time.time() - t0
        print(f"epoch {epoch}: " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()), flush=True)
        if ml is not None:
            ml.log_metrics(metrics, step=epoch)
        path = os.path.join(a.save_dir, f"wm_epoch_{epoch:03d}.pth")
        torch.save({"model": model.state_dict(), "target": target.state_dict(), "model_config": dataclasses.asdict(mcfg.resolved()),
                    "run_config": run_config, "val_routes": val_routes, "metrics": metrics, "epoch": epoch}, path + ".tmp")
        os.replace(path + ".tmp", path)
    if ml is not None:
        ml.end_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
