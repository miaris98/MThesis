"""Mark which DAgger frames carry a valid expert label (idea 19e / T5, 2026-11-08).

A saved frame f (tick f * data_save_freq) is labelled by the car's own poses over the next pred_len saved frames
(wor_dataset._index_pdm_lite_route). It is a valid expert correction only if the expert controlled the car on every tick from f's tick to
f's tick + KEEP_HORIZON_TICKS (default 40 = 2 s: the 25-tick label horizon plus margin for the anticipation targets). Writes
"dagger_keep": true/false into every measurements/<frame>.json.gz of each route folder that has dagger_sources.json.gz, and prints counts.

    python scripts/datagen/dagger_prune.py ROOT [--horizon_ticks 40]
"""
import argparse
import glob
import gzip
import json
import os


def prune_route(route_dir, horizon):
    with gzip.open(os.path.join(route_dir, "dagger_sources.json.gz"), "rt", encoding="utf-8") as f:
        d = json.load(f)
    freq = int(d["data_save_freq"])
    src = {int(s): k for s, k in d["sources"]}
    kept = total = 0
    for mf in sorted(glob.glob(os.path.join(route_dir, "measurements", "*.json.gz"))):
        frame = int(os.path.basename(mf).split(".")[0])
        t0 = frame * freq
        keep = all(src.get(t) == "expert" for t in range(t0, t0 + horizon + 1))
        with gzip.open(mf, "rt", encoding="utf-8") as f:
            m = json.load(f)
        m["dagger_keep"] = bool(keep)
        with gzip.open(mf, "wt", encoding="utf-8") as f:
            json.dump(m, f)
        kept += keep
        total += 1
    return kept, total, int(d["takeovers"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--horizon_ticks", type=int, default=40)
    a = ap.parse_args()
    K = T = TO = n = 0
    for sf in sorted(glob.glob(os.path.join(a.root, "**", "dagger_sources.json.gz"), recursive=True)):
        k, t, to = prune_route(os.path.dirname(sf), a.horizon_ticks)
        print(f"{os.path.dirname(sf)}: takeovers {to}, kept {k} of {t} frames")
        K, T, TO, n = K + k, T + t, TO + to, n + 1
    print(f"DAGGER_PRUNE_DONE routes {n} takeovers {TO} kept {K} of {T} frames")


if __name__ == "__main__":
    main()
