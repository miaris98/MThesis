"""Train the object detector on the frozen RGB features (2026-11-18, step 2 of the teacher ladder, stage A: how well can our frozen
encoder's map locate the objects the teacher drives from?).

Inputs, all from one pooled cache (build_pooled_cache.py ... --pca_full 256): the PCA-256 full map <prefix>.pca256.f16 (N, 256, 9, 24) and
the camera-visible teacher tokens <prefix>.boxtok20_visns.f16 (build_box_targets.py --box_variant _visns) as targets. Routes are split
train / val (route folders, seeded), never frames of one route on both sides.

Reports per validation pass: recall, precision and their F1 (detector_best.pt is the best F1 by default, --select) at a 2 m centre tolerance, position error of matched objects, both by distance bin
(0-10 / 10-20 / 20-30 / 30-50 m) and recall by class. Logs to MLflow (paths.mlruns_dir(), MLFLOW_ALLOW_FILE_STORE=true). Saves
detector_best.pt / detector_last.pt and metrics.json in --out_dir.

    python scripts/training/train_object_detector.py --prefix /workspace/pooled_ov0/<prefix> --out_dir /workspace/checkpoints/det_visns
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.training.build_box_targets import BOX_FEATS  # noqa: E402
from src.models.world_on_rails.object_detector import DetectionMeter, ObjectDetector, detector_loss  # noqa: E402


def route_of(key: str) -> str:
    """'Town01/data/Scenario/route_x/rgb/0001.jpg' -> 'Town01/data/Scenario/route_x' (rgb_augmented frames belong to the same route)."""
    return os.path.dirname(os.path.dirname(key))


class CacheRows(Dataset):
    def __init__(self, prefix: str, rows: np.ndarray, pca: int, variant: str, m: int):
        meta = json.load(open(prefix + ".json"))
        self.n = meta["n"]
        self.pca_path = f"{prefix}.pca{pca}.f16"
        self.pca_shape = tuple(json.load(open(self.pca_path + ".json"))["shape"])
        self.tok_path = f"{prefix}.boxtok{m}{variant}.f16"
        self.m, self.rows = m, rows
        self.pm = self.tm = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.pm is None:   # opened per worker
            self.pm = np.memmap(self.pca_path, dtype=np.float16, mode="r", shape=self.pca_shape)
            self.tm = np.memmap(self.tok_path, dtype=np.float16, mode="r", shape=(self.n, self.m, BOX_FEATS))
        r = int(self.rows[i])
        return torch.from_numpy(np.array(self.pm[r])), torch.from_numpy(np.array(self.tm[r], dtype=np.float32))


def split_rows(prefix: str, val_split: float, seed: int):
    meta = json.load(open(prefix + ".json"))
    done = np.memmap(prefix + ".done", dtype=np.uint8, mode="r", shape=(meta["n"],))
    routes = sorted({route_of(k) for k in meta["keys"]})
    rng = random.Random(seed)
    rng.shuffle(routes)
    val_routes = set(routes[: max(1, int(round(val_split * len(routes))))])
    is_val = np.array([route_of(k) in val_routes for k in meta["keys"]])
    ok = np.asarray(done, dtype=bool)
    return np.where(ok & ~is_val)[0], np.where(ok & is_val)[0], len(routes), len(val_routes)


@torch.no_grad()
def evaluate(det, loader, device, max_batches=None):
    det.eval()
    meter, tot, n = DetectionMeter(), 0.0, 0
    for bi, (x, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = det(x)
        loss, _ = detector_loss(out, y)
        tot += float(loss) * len(x); n += len(x)
        meter.update(out, y)
    det.train()
    s = meter.summary()
    s["loss"] = tot / max(n, 1)
    s["f1"] = 2 * s["recall"] * s["precision"] / max(s["recall"] + s["precision"], 1e-9)   # the checkpoint-selection metric (--select f1)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant", default="_visns", help="token file the targets come from (the camera-visible objects, speed zeroed)")
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--val_batches", type=int, default=200, help="validation batches per pass (the full val set at the end)")
    ap.add_argument("--max_minutes", type=float, default=120.0)
    ap.add_argument("--select", choices=("f1", "loss"), default="f1",
                    help="detector_best.pt = highest val F1 of recall/precision at the 2 m tolerance (default) or lowest val loss. "
                         "The loss rises once objectness gets over-confident while matching keeps improving (2026-11-18).")
    ap.add_argument("--max_batches", type=int, default=0, help="smoke test: stop each epoch after this many batches")
    ap.add_argument("--use_mlflow", type=int, default=1)
    ap.add_argument("--experiment_name", default="teacher_ladder")
    ap.add_argument("--run_label", default="detector_visns")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_rows, va_rows, n_routes, n_val = split_rows(a.prefix, a.val_split, a.seed)
    print(f"--> {len(tr_rows)} train / {len(va_rows)} val frames ({n_routes} routes, {n_val} held out), targets {a.variant}", flush=True)
    mk = lambda rows, sh: DataLoader(CacheRows(a.prefix, rows, a.pca, a.variant, a.m), batch_size=a.batch_size, shuffle=sh, drop_last=sh,
                                     num_workers=a.num_workers, pin_memory=True, prefetch_factor=2 if a.num_workers else None,
                                     persistent_workers=bool(a.num_workers))
    tr, va = mk(tr_rows, True), mk(va_rows, False)
    det = ObjectDetector(in_ch=a.pca, dim=a.dim, queries=a.queries, layers=a.layers).to(device)
    opt = torch.optim.AdamW(det.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * (min(len(tr), a.max_batches) if a.max_batches else len(tr))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / max(steps, 1))))
    cfg = dict(vars(a)); cfg.update(train_frames=len(tr_rows), val_frames=len(va_rows), routes=n_routes, val_routes=n_val)
    json.dump(cfg, open(os.path.join(a.out_dir, "run_config.json"), "w"), indent=2)
    ml = None
    if a.use_mlflow:
        try:
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            import mlflow
            try:
                from src.utils import paths
                mlflow.set_tracking_uri(f"file:{paths.mlruns_dir()}")
            except Exception:
                pass
            mlflow.set_experiment(a.experiment_name)
            ml = mlflow.start_run(run_name=a.run_label)
            mlflow.log_params({k: v for k, v in cfg.items() if isinstance(v, (int, float, str))})
        except Exception as e:
            print(f"[mlflow] off: {e!r}", flush=True); ml = None
    t0, best, step = time.time(), None, 0
    for ep in range(1, a.epochs + 1):
        run = {"loss": 0.0, "n": 0}
        for bi, (x, y) in enumerate(tr):
            if a.max_batches and bi >= a.max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = det(x)
            loss, parts = detector_loss(out, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(det.parameters(), 1.0); opt.step(); sched.step(); step += 1
            run["loss"] += float(loss); run["n"] += 1
            if step % 200 == 0:
                print(f"    epoch {ep} step {step} loss {run['loss'] / run['n']:.4f} {parts} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                if ml:
                    import mlflow; mlflow.log_metrics({"train_loss": run["loss"] / run["n"], **{f"train_{k}": v for k, v in parts.items()}}, step=step)
            if (time.time() - t0) / 60 > a.max_minutes:
                break
        s = evaluate(det, va, device, a.val_batches)
        print(f"--> epoch {ep:02d} train loss {run['loss'] / max(run['n'], 1):.4f} | val loss {s['loss']:.4f} recall {s['recall']:.3f} "
              f"precision {s['precision']:.3f} pos_err {s['pos_err_m']:.2f} m cls_acc {s['cls_acc']:.3f} | "
              + " ".join(f"{k} {v:.3f}" for k, v in s.items() if k.startswith("recall_") and "m" in k.split("_")[-1]), flush=True)
        if ml:
            import mlflow; mlflow.log_metrics({f"val_{k}": float(v) for k, v in s.items()}, step=step)
        torch.save({"state_dict": det.state_dict(), "config": cfg, "epoch": ep, "val": {k: float(v) for k, v in s.items()}}, os.path.join(a.out_dir, "detector_last.pt"))
        score = s["f1"] if a.select == "f1" else -s["loss"]
        if best is None or score > best:   # 2026-11-18: val loss bottomed at epoch 2 while recall/precision rose to epoch 4
            best = score
            torch.save({"state_dict": det.state_dict(), "config": cfg, "epoch": ep, "val": {k: float(v) for k, v in s.items()}}, os.path.join(a.out_dir, "detector_best.pt"))
        if (time.time() - t0) / 60 > a.max_minutes:
            print(f"--> stopped at the {a.max_minutes:.0f}-minute cap", flush=True)
            break
    det.load_state_dict(torch.load(os.path.join(a.out_dir, "detector_best.pt"), map_location=device, weights_only=False)["state_dict"])
    s = evaluate(det, va, device)   # the full validation set
    json.dump({k: float(v) for k, v in s.items()}, open(os.path.join(a.out_dir, "metrics.json"), "w"), indent=2)
    print("FINAL " + json.dumps({k: round(float(v), 4) for k, v in s.items()}), flush=True)
    if ml:
        import mlflow; mlflow.log_metrics({f"final_{k}": float(v) for k, v in s.items()}); mlflow.log_artifact(os.path.join(a.out_dir, "metrics.json")); mlflow.end_run()


if __name__ == "__main__":
    main()
