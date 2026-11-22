"""Export the R1 detector tokens (2026-11-19, teacher ladder step 2): run the frozen object detector once over every cached frame's PCA map
and write teacher-format tokens next to the pooled cache, so policy training reads 20 x 16 numbers per frame instead of the 9 x 24 x 256 map.

    python scripts/training/export_detector_tokens.py --prefix /workspace/pooled_ov0/<p>,/workspace/pooled_cov_ov0/<p> \
        --detector /workspace/checkpoints/det_visns/detector_best.pt --m 20 --variant _det --max_range 30

Per prefix it writes <prefix>.boxtok<m><variant>.f16 (N, m, BOX_FEATS) and a .json with the settings; train_wor.py --object_tokens refuses
a token file whose max_range / threshold differ from the run's. Every prefix must hold its PCA map in the SAME basis the detector was trained
on (the policy runs that basis live): a cache that fitted its own basis is refused. Rows the cache never encoded stay all-zero (no objects).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.training.build_box_targets import BOX_FEATS, VALID_SLOT  # noqa: E402
from src.models.world_on_rails.object_detector import ObjectDetector  # noqa: E402
from src.training.pooled_feature_cache import pca_file  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def same_basis(a: str, b: str) -> bool:
    if os.path.abspath(a) == os.path.abspath(b):
        return True
    x, y = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu")
    return torch.equal(x["basis"], y["basis"]) and torch.equal(x["mean"], y["mean"])


def load_detector(path: str, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    det = ObjectDetector(in_ch=int(cfg.get("pca", 256)), dim=int(cfg.get("dim", 256)), queries=int(cfg.get("queries", 30)),
                         layers=int(cfg.get("layers", 3)))
    det.load_state_dict(ck["state_dict"])
    return det.to(device).eval(), cfg


@torch.no_grad()
def export(prefix: str, det, cfg: dict, det_path: str, a, device) -> dict:
    K = int(cfg.get("pca", 256))
    pf = pca_file(prefix, K)
    pmeta = json.load(open(pf + ".json"))
    basis = pmeta.get("basis") or pf + ".basis.pt"
    det_basis = pca_file(cfg["prefix"], K) + ".basis.pt" if cfg.get("prefix") else None
    if det_basis and os.path.exists(det_basis) and not same_basis(basis, det_basis):
        raise SystemExit(f"{pf} was compressed with another PCA basis than the detector's ({det_basis}): rebuild it with that basis "
                         f"(copy the .basis.pt next to it before build_pooled_cache.py --pca_full)")
    N = int(pmeta["shape"][0])
    pm = np.memmap(pf, dtype=np.float16, mode="r", shape=tuple(pmeta["shape"]))
    done_path = prefix + ".done"
    done = np.memmap(done_path, dtype=np.uint8, mode="r", shape=(N,)) if os.path.exists(done_path) else np.ones(N, np.uint8)
    out_path = f"{prefix}.boxtok{a.m}{a.variant}.f16"
    out = np.memmap(out_path + ".tmp", dtype=np.float16, mode="w+", shape=(N, a.m, BOX_FEATS))
    rows = np.where(np.asarray(done, dtype=bool))[0]
    t0, nvalid = time.time(), 0
    for i in range(0, len(rows), a.batch_size):
        r = rows[i:i + a.batch_size]
        x = torch.from_numpy(np.asarray(pm[r], dtype=np.float32)).to(device)   # the cache's f16 values, as the live path rounds them
        tok = det.tokens(det(x), m=a.m, threshold=a.threshold, max_range=a.max_range)
        out[r] = tok.cpu().numpy().astype(np.float16)
        nvalid += int(tok[..., VALID_SLOT].sum())
        if (i // a.batch_size) % 100 == 0:
            print(f"    {prefix}: {i + len(r)}/{len(rows)} frames ({time.time() - t0:.0f} s)", flush=True)
    out.flush()
    del out
    os.replace(out_path + ".tmp", out_path)
    meta = {"n": N, "m": a.m, "variant": a.variant, "max_range": a.max_range, "threshold": a.threshold, "detector": det_path,
            "detector_sha256": sha256(det_path), "basis": basis, "frames_exported": int(len(rows)),
            "mean_objects_per_frame": nvalid / max(len(rows), 1)}
    json.dump(meta, open(out_path + ".json", "w"), indent=2)
    print(f"--> {out_path}: {len(rows)} frames, {meta['mean_objects_per_frame']:.2f} objects per frame ({time.time() - t0:.0f} s)", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="pooled-cache prefix(es), comma-separated")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--variant", default="_det")
    ap.add_argument("--max_range", type=float, default=30.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=512)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det, cfg = load_detector(a.detector, device)
    for p in [p for p in a.prefix.split(",") if p]:
        export(p, det, cfg, a.detector, a, device)


if __name__ == "__main__":
    main()
