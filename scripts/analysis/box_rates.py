#!/usr/bin/env python3
"""Measured rates for planning (S-124, 2026-10-04): how fast do our boxes actually go?

Everything is read from files already on E: (nothing is run):

* **CARLA evaluation throughput** per box: the `save_name` of every Leaderboard record ends in the run's start time (`..._<weather>_MM_DD_HH_MM_SS`), so the number of
  route-runs per hour of a box (12 lanes) comes from the first and last start of its result files, plus the busiest 3 h window (steady state).
* **CARLA training minutes per epoch** from the mtimes of consecutive `model_epoch_*.pth` of every arm (tar -x keeps mtimes), with the arm's image size.
* **Atari training minutes per 1k environment steps** from the mtimes of the `checkpoint_env<N>_upd<M>.pt` files (one run per GPU vs several per GPU matters: read the box).

    py scripts/analysis/box_rates.py [E:/MThesis_EXP]
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import re
import statistics
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "E:/MThesis_EXP"
START = re.compile(r"_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})$")


def norm(paths):
    return [p.replace("\\", "/") for p in paths]


def carla_eval():
    print("== CARLA evaluation: route-runs per hour per box (12 lanes) ==")
    for box in sorted(norm(glob.glob(f"{ROOT}/live_2026100*_*carla*"))):
        runs = []
        for f in norm(glob.glob(box + "/**/*_x.json", recursive=True)):
            try:
                recs = json.load(open(f))["_checkpoint"]["records"]
            except Exception:
                continue
            for r in recs:
                m = START.search(r["save_name"])
                if m:
                    mo, d, h, mi, s = map(int, m.groups())
                    runs.append((dt.datetime(2026, mo, d, h, mi, s), r["meta"].get("duration_system")))
        if len(runs) < 50:
            continue
        runs.sort()
        span = (runs[-1][0] - runs[0][0]).total_seconds() / 3600
        best, j = 0, 0
        for i in range(len(runs)):
            while (runs[i][0] - runs[j][0]).total_seconds() > 3 * 3600:
                j += 1
            best = max(best, i - j + 1)
        wall = [x[1] for x in runs if x[1]]
        print(f"{os.path.basename(box)[:44]:44s} {len(runs):4d} runs in {span:4.1f} h = {len(runs) / max(span, 0.1):5.0f} runs/h; busiest 3 h: {best / 3:4.0f} runs/h; "
              f"median wall per run {statistics.median(wall):4.0f} s")


def carla_train():
    print("\n== CARLA training: minutes per epoch (frozen backbone, live encoder, no feature cache) ==")
    rows = []
    for d in norm(glob.glob(f"{ROOT}/live_*/checkpoints/carla_arm*")):
        t = []
        for f in glob.glob(d + "/model_epoch_*.pth"):
            t.append((int(re.search(r"epoch_(\d+)", f).group(1)), os.path.getmtime(f)))
        t.sort()
        deltas = [(b[1] - a[1]) / 60 for a, b in zip(t, t[1:]) if b[0] - a[0] == 1]
        if len(deltas) < 3:
            continue
        try:
            cfg = json.load(open(d + "/run_config.json"))
        except Exception:
            cfg = {}
        rows.append((d.split("/live_")[1].split("/")[0][:30], os.path.basename(d)[:26], len(deltas), statistics.median(deltas), cfg.get("img_size"), cfg.get("feature_cache_tag")))
    for r in sorted(rows):
        print(f"{r[0]:30s} {r[1]:26s} {r[2]:2d} epochs, median {r[3]:5.1f} min/epoch, img {r[4]}, feature cache {r[5]}")


def atari_train():
    print("\n== Atari (EZ-V2 port) training: minutes per 1k env steps ==")
    rows = []
    for d in norm(glob.glob(f"{ROOT}/live_*/**/checkpoints", recursive=True)):
        cks = []
        for f in glob.glob(d + "/checkpoint_env*_upd*.pt"):
            m = re.search(r"checkpoint_env(\d+)_upd(\d+)\.pt$", f)
            if m:
                cks.append((int(m.group(1)), os.path.getmtime(f)))
        cks.sort()
        if len(cks) < 4 or "hf_checkpoints" in d:
            continue
        steps = cks[-1][0] - cks[0][0]
        span_h = (cks[-1][1] - cks[0][1]) / 3600
        if steps <= 0 or span_h < 0.5:
            continue
        run = d.split("S058_ezv2_match/")[-1].split("/checkpoints")[0]
        rows.append((d.split("/live_")[1].split("/")[0][:30], run[:36], cks[0][0], cks[-1][0], span_h * 60 / (steps / 1000)))
    for r in sorted(rows):
        print(f"{r[0]:30s} {r[1]:36s} env {r[2]:6d} -> {r[3]:6d}: {r[4]:5.1f} min per 1k steps")


if __name__ == "__main__":
    carla_eval()
    carla_train()
    atari_train()
