"""TODO A84 (2026-11-16): write the expert-reason / danger / bucket labels (src/training/reason_labels.py) for every PDM-Lite route of a data dir.

CPU only, no simulator, no images: one process per route. Each route gets `<route>/reason_<tag>.npz`; train_wor.py reads it with
`--reason_labels <tag>`. The printed summary is the step-0 count asked for on 2026-11-16 before any training money is spent:

  * cause shares       which object binds the expert's speed (none / vehicle / vulnerable / light / stop / static), and on how many frames;
  * bucket shares      what the re-balancing sampler would draw from;
  * danger rate        per scenario folder: how often proceeding at max(speed, 3 m/s) for 2 s would hit someone (the crash-classifier label);
  * label noise        on danger frames the expert should slow down or stay slow: `slow_went` (standing, yet the expert drives off within 1 s)
                       and `moving_kept` (moving, yet the expert does not slow within 1 s) should both stay small;
  * coverage           frames labelled / frames read (box files missing or unreadable leave -1).

    python scripts/training/build_reason_labels.py --data_dir /workspace/dataset/wor_trajectories --tag v1 [--workers 32] [--limit 5]
"""
import argparse
import collections
import json
import os
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one process per route: BLAS threads would oversubscribe the cores
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scripts.training.build_q_labels import _read, _usable_cpus, find_routes  # noqa: E402
from src.training.reason_labels import (BUCKETS, REASON_CLASSES, ReasonConfig, expert_consistency, label_path,  # noqa: E402
                                        label_route, save_labels)


def label_one(args):
    route_dir, tag, overwrite = args
    out_path = label_path(route_dir, tag)
    cfg = ReasonConfig()
    t0 = time.time()
    meas_dir = os.path.join(route_dir, "measurements")
    ids = sorted(f.split(".")[0] for f in os.listdir(meas_dir) if f.endswith(".json.gz"))
    meas = [_read(os.path.join(meas_dir, f"{fid}.json.gz")) for fid in ids]
    boxes = [_read(os.path.join(route_dir, "boxes", f"{fid}.json.gz")) for fid in ids]
    if os.path.exists(out_path) and not overwrite:
        from src.training.reason_labels import load_labels
        _, cols = load_labels(out_path, cfg)     # raises on a config mismatch, so a stale file is never counted as done
        labels = cols
    else:
        labels = label_route(meas, boxes, cfg)
        save_labels(out_path, ids, labels, cfg)
    scen = os.path.basename(os.path.dirname(route_dir.rstrip("/\\")))
    return {
        "route": route_dir, "scenario": scen, "frames": len(ids), "unreadable": sum(m is None for m in meas),
        "no_boxes": sum(b is None for b in boxes),
        "cause": np.bincount(labels["cause"][labels["cause"] >= 0], minlength=len(REASON_CLASSES)).tolist(),
        "bucket": np.bincount(labels["bucket"][labels["bucket"] >= 0], minlength=len(BUCKETS)).tolist(),
        "danger_pos": int((labels["danger"] == 1).sum()), "danger_valid": int((labels["danger"] >= 0).sum()),
        "brake_pos": int((labels["brake"] == 1).sum()), "brake_valid": int((labels["brake"] >= 0).sum()),
        "noise": expert_consistency(meas, labels, cfg), "sec": time.time() - t0,
    }


def summarise(results, out=sys.stdout):
    tot = collections.Counter()
    cause = np.zeros(len(REASON_CLASSES), int)
    bucket = np.zeros(len(BUCKETS), int)
    noise = collections.Counter()
    by_scen = collections.defaultdict(lambda: [0, 0, 0])      # danger positives, danger valid, frames
    for r in results:
        tot.update(frames=r["frames"], unreadable=r["unreadable"], no_boxes=r["no_boxes"], danger_pos=r["danger_pos"],
                   danger_valid=r["danger_valid"], brake_pos=r["brake_pos"], brake_valid=r["brake_valid"])
        cause += np.asarray(r["cause"])
        bucket += np.asarray(r["bucket"])
        noise.update(r["noise"])
        s = by_scen[r["scenario"]]
        s[0] += r["danger_pos"]
        s[1] += r["danger_valid"]
        s[2] += r["frames"]
    f = max(1, tot["frames"])
    print(f"routes {len(results)}  frames {tot['frames']}  unreadable {tot['unreadable']}  without boxes {tot['no_boxes']}", file=out)
    print("cause (binding object):  " + "  ".join(f"{n} {c} ({100 * c / max(1, cause.sum()):.1f}%)" for n, c in zip(REASON_CLASSES, cause)), file=out)
    print("buckets:                 " + "  ".join(f"{n} {c} ({100 * c / max(1, bucket.sum()):.1f}%)" for n, c in zip(BUCKETS, bucket)), file=out)
    print(f"brake within 1 s: {tot['brake_pos']}/{tot['brake_valid']} ({100 * tot['brake_pos'] / max(1, tot['brake_valid']):.1f}%)", file=out)
    print(f"danger: {tot['danger_pos']}/{tot['danger_valid']} ({100 * tot['danger_pos'] / max(1, tot['danger_valid']):.1f}%) of labelled frames; "
          f"coverage {100 * tot['danger_valid'] / f:.1f}% of frames", file=out)
    sd, mv = max(1, noise["slow_danger"]), max(1, noise["moving_danger"])
    print(f"label noise: standing/slow danger frames where the expert drives off within 1 s {noise['slow_went']}/{noise['slow_danger']} "
          f"({100 * noise['slow_went'] / sd:.1f}%); moving danger frames where it does not slow {noise['moving_kept']}/{noise['moving_danger']} "
          f"({100 * noise['moving_kept'] / mv:.1f}%)", file=out)
    print("danger rate by scenario folder:", file=out)
    for k, (p, v, n) in sorted(by_scen.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][1])):
        print(f"  {k:42s} {p:6d}/{v:6d} ({100 * p / max(1, v):5.1f}%)  frames {n}", file=out)
    return {"routes": len(results), **tot, "cause": dict(zip(REASON_CLASSES, cause.tolist())), "bucket": dict(zip(BUCKETS, bucket.tolist())),
            "noise": dict(noise), "danger_by_scenario": {k: v[:2] for k, v in by_scen.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_dir", required=True, help="comma-separated PDM-Lite roots (searched recursively for <route>/measurements + boxes)")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--workers", type=int, default=0, help="0 = the container's CPU quota")
    ap.add_argument("--limit", type=int, default=0, help="label only the first N routes (smoke test)")
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--stats_out", default="", help="write the summary as JSON here")
    ap.add_argument("--use_mlflow", type=int, default=1)
    args = ap.parse_args()
    routes = [r for d in args.data_dir.split(",") for r in find_routes(d)]
    if args.limit:
        routes = routes[:args.limit]
    if not routes:
        raise SystemExit(f"no PDM-Lite routes (measurements/ + boxes/) under {args.data_dir}")
    workers = args.workers or _usable_cpus()
    print(f"--> labelling {len(routes)} routes with {workers} workers, tag {args.tag}, config {ReasonConfig().to_json()}", flush=True)
    t0, results = time.time(), []
    jobs = [(r, args.tag, bool(args.overwrite)) for r in routes]
    if workers > 1:
        with Pool(workers) as pool:
            for k, res in enumerate(pool.imap_unordered(label_one, jobs, chunksize=1), 1):
                results.append(res)
                if k % 200 == 0:
                    print(f"    {k}/{len(routes)} routes, {time.time() - t0:.0f} s", flush=True)
    else:
        results = [label_one(j) for j in jobs]
    stats = summarise(results)
    stats.update(tag=args.tag, config=json.loads(ReasonConfig().to_json()), seconds=round(time.time() - t0, 1))
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=1)
    if args.use_mlflow:   # CLAUDE.md: label passes are tracked like runs (params = config, metrics = the step-0 numbers)
        try:
            import mlflow
            from src.config.paths import mlruns_dir
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:" + str(mlruns_dir())))   # same store as build_q_labels.py
            mlflow.set_experiment("A84_reason_labels")
            with mlflow.start_run(run_name=f"reason_labels_{args.tag}"):
                mlflow.log_params({f"cfg_{k}": v for k, v in stats["config"].items()})
                mlflow.log_metrics({"routes": stats["routes"], "frames": stats["frames"],
                                    "danger_rate": stats["danger_pos"] / max(1, stats["danger_valid"]),
                                    "brake_rate": stats["brake_pos"] / max(1, stats["brake_valid"])})
                if args.stats_out:
                    mlflow.log_artifact(args.stats_out)
        except Exception as exc:     # tracking must never cost the labels
            print(f"[Warning] MLflow logging skipped: {exc}", flush=True)
    print(f"--> done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
