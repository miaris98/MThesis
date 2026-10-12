#!/usr/bin/env python3
"""TODO A36: weight-space merge of two heads-only CARLA checkpoints that share one frozen backbone.

theta(alpha) = (1 - alpha) * theta_A + alpha * theta_B on the trained tensors (the `model` entry of the heads-only
`model_epoch_*.pth`; the frozen backbone is not touched). With several --b checkpoints the B side is their mean (a "soup").
The output directory looks like a training run's save dir, so the eval lanes can use it unchanged:

    <out>/model_epoch_<EPOCH>.pth     merged heads (config, partial / frozen_ref copied from the first B checkpoint; no optimizer state)
    <out>/frozen_backbone.pth         copied from the first B checkpoint, after checking that A's backbone tensors are identical
    <out>/run_config.json             copied from the first B checkpoint's directory
    <out>/MERGE.json                  what was merged, alpha, tensor counts, max |diff| to the endpoints, sha256 of the files

Checks that stop the merge: backbone tensors differ, key sets or shapes differ, any non-finite value. The test mode
(`--selftest`) merges at alpha 0 and 1 and requires exact equality with the endpoints.

    py scripts/training/a36_merge_checkpoints.py --a E:/.../carla_armE_aug1_hires/model_epoch_015.pth \
        --b E:/.../carla_armJ_ft_obst/model_epoch_020.pth --alpha 0.5 --out E:/MThesis_EXP/prep_20261005/a36/alpha0.50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_heads(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not ck.get("partial"):
        raise SystemExit(f"{path}: not a heads-only checkpoint (partial flag missing); merge the heads only")
    return ck


def backbone_state(ckpt_path: Path) -> dict:
    ref = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("frozen_ref", "frozen_backbone.pth")
    fb = torch.load(ckpt_path.parent / ref, map_location="cpu", weights_only=False)
    return fb.get("model", fb)


def same_tensors(a: dict, b: dict) -> tuple[bool, str]:
    if a.keys() != b.keys():
        return False, f"key sets differ ({len(a)} vs {len(b)})"
    for k in a:
        if a[k].shape != b[k].shape or not torch.equal(a[k], b[k]):
            return False, f"tensor {k} differs"
    return True, ""


def merge(a_sd: dict, b_sds: list, alpha: float) -> dict:
    out = {}
    for k, va in a_sd.items():
        vb = sum(sd[k].double() for sd in b_sds) / len(b_sds) if va.is_floating_point() else b_sds[0][k]
        out[k] = ((1 - alpha) * va.double() + alpha * vb).to(va.dtype) if va.is_floating_point() else vb
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="checkpoint at alpha = 0 (e.g. arm E e15)")
    ap.add_argument("--b", required=True, nargs="+", help="checkpoint(s) at alpha = 1 (e.g. arm J e20; several = their mean)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epoch", type=int, default=None, help="epoch number in the output file name (default: the first B checkpoint's)")
    ap.add_argument("--selftest", action="store_true", help="also merge at alpha 0 and 1 and require exact equality with the endpoints")
    a = ap.parse_args()

    pa, pbs = Path(a.a), [Path(x) for x in a.b]
    ca, cbs = load_heads(pa), [load_heads(p) for p in pbs]
    ok, why = same_tensors(backbone_state(pa), backbone_state(pbs[0]))
    if not ok:
        raise SystemExit(f"frozen backbones of A and B differ ({why}): the heads are not in one basin / on one backbone, refusing to merge")
    for p, c in zip(pbs, cbs):
        ok, why = same_tensors(backbone_state(pbs[0]), backbone_state(p))
        if not ok:
            raise SystemExit(f"frozen backbones of the B checkpoints differ ({why})")
    a_sd, b_sds = ca["model"], [c["model"] for c in cbs]
    for p, sd in zip(pbs, b_sds):
        if a_sd.keys() != sd.keys():
            raise SystemExit(f"{p}: key set differs from A ({len(a_sd)} vs {len(sd)} tensors)")
        bad = [k for k in a_sd if a_sd[k].shape != sd[k].shape]
        if bad:
            raise SystemExit(f"{p}: shape mismatch for {bad[:3]}")

    merged = merge(a_sd, b_sds, a.alpha)
    nonfinite = [k for k, v in merged.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    if nonfinite:
        raise SystemExit(f"non-finite values in {nonfinite[:3]}")

    if a.selftest:
        for alpha, ref in ((0.0, a_sd), (1.0, {k: sum(sd[k].double() for sd in b_sds).to(a_sd[k].dtype) / len(b_sds) if a_sd[k].is_floating_point() else b_sds[0][k] for k in a_sd})):
            m = merge(a_sd, b_sds, alpha)
            worst = max((m[k].double() - ref[k].double()).abs().max().item() for k in a_sd if a_sd[k].is_floating_point())
            print(f"selftest alpha={alpha}: max |merged - endpoint| = {worst:.3e}")
            if worst > 1e-6:
                raise SystemExit("selftest failed")

    epoch = a.epoch if a.epoch is not None else int(cbs[0].get("epoch", 0))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = {k: v for k, v in cbs[0].items() if k not in ("model", "optimizer", "metrics")}
    ck["model"] = merged
    ck["epoch"] = epoch
    ck["merge"] = {"a": str(pa), "b": [str(p) for p in pbs], "alpha": a.alpha}
    fname = out / f"model_epoch_{epoch:03d}.pth"
    tmp = fname.with_suffix(".tmp")
    torch.save(ck, tmp)
    tmp.replace(fname)
    shutil.copy2(pbs[0].parent / ck.get("frozen_ref", "frozen_backbone.pth"), out / "frozen_backbone.pth")
    cfg = pbs[0].parent / "run_config.json"
    if cfg.exists():
        shutil.copy2(cfg, out / "run_config.json")
    fl = [k for k, v in merged.items() if v.is_floating_point()]
    rel = {"to_A": max((merged[k].double() - a_sd[k].double()).abs().max().item() for k in fl),
           "to_B": max((merged[k].double() - b_sds[0][k].double()).abs().max().item() for k in fl)}
    info = {"a": str(pa), "b": [str(p) for p in pbs], "alpha": a.alpha, "tensors": len(merged), "float_tensors": len(fl),
            "max_abs_diff": rel, "files": {p.name: sha256(p) for p in (fname, out / "frozen_backbone.pth")}}
    (out / "MERGE.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {fname} ({len(merged)} tensors, alpha {a.alpha}); max |merged - A| {rel['to_A']:.3e}, |merged - B| {rel['to_B']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
