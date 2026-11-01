"""TODO A71 (2026-11-01): write the counterfactual Q labels (src/training/counterfactual_q.py) for every PDM-Lite route of a data dir.

CPU only, no simulator: one process per route (multiprocessing, all cores by default). Each route gets `<route>/<tag>.npz` with
`frame` (frame ids), `q` (n_frames, 8 speed bins, 5 lateral offsets; NaN where a frame has fewer than `horizon` logged successors) and `config`
(the QConfig as JSON). train_wor.py reads them with `--q_critic 1 --q_labels <tag>`.

Sanity numbers printed and logged to MLflow, before any training:
  * expert_top3_share - how often the expert's own action (speed bin nearest its target speed, lateral offset nearest its route shift) is among
    the 3 best of the 40 actions; low values mean the reward disagrees with the expert and the labels would pull the policy away from it;
  * expert_collision_share - how often the expert's own action collides in its best sequence (true hit flags, not a Q threshold); near 0;
  * safe_action_share - frames where at least one action is collision-free; near 1 (else the scene geometry is wrong);
  * spread - mean (max - min) of Q per frame; near 0 means the labels carry no signal;
  * distinct_q - mean number of distinct Q values per frame (of 40); low values mean the action grid collapses.

    python scripts/training/build_q_labels.py --data_dir /workspace/dataset/wor_trajectories --tag qcf_h12 [--workers 32] [--limit 5]
"""
import argparse
import dataclasses
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one process per route: BLAS threads would oversubscribe the cores
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.training.counterfactual_q import QConfig, label_route  # noqa: E402


def _usable_cpus() -> int:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:                                                         # cgroup v2 quota ("max 100000" = unlimited)
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except Exception:
        pass
    return max(1, n)


def find_routes(data_dir: str):
    out = []
    for root, dirs, _ in os.walk(data_dir):
        if "measurements" in dirs and "boxes" in dirs:
            out.append(root)
            dirs[:] = []
    return sorted(out)


def _read(path):
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception:
        return None


def expert_action_index(meas: dict, cfg: QConfig):
    """(speed index, lateral index) of the expert's own choice: its target speed, and the shift of its route off route_original ~5-10 m ahead."""
    ts = float(meas.get("target_speed", meas.get("speed", 0.0)))
    v = int(np.argmin(np.abs(np.array(cfg.speed_targets) - ts)))
    shift = 0.0
    r, ro = meas.get("route"), meas.get("route_original")
    if r and ro and len(r) >= 8 and len(ro) >= 8:
        shift = float(r[0][1] - ro[0][1]) if ts < 1.0 else float(np.mean([r[j][1] - ro[j][1] for j in range(4, 8)]))
    d = int(np.argmin(np.abs(np.array(cfg.lat_targets) - shift)))
    return v, d


def label_route_dir(route_dir: str, tag: str, cfg: QConfig, overwrite: bool = False) -> dict:
    out_path = os.path.join(route_dir, f"{tag}.npz")
    if os.path.exists(out_path) and not overwrite:
        with np.load(out_path, allow_pickle=False) as z:
            if "config" in z.files and json.loads(str(z["config"])) != json.loads(json.dumps(dataclasses.asdict(cfg))):
                raise ValueError(f"{out_path} was built with another config: pass --overwrite or use a new --tag")
        return {"route": route_dir, "skipped": True}
    t0 = time.time()
    names = sorted(n for n in os.listdir(os.path.join(route_dir, "measurements")) if n.endswith(".json.gz"))
    frames = []
    for n in names:
        m, b = _read(os.path.join(route_dir, "measurements", n)), _read(os.path.join(route_dir, "boxes", n))
        frames.append((m, b) if m is not None and b is not None and "ego_matrix" in m else None)
    q, ok, hit = label_route(frames, cfg)
    ids = np.array([n.split(".")[0] for n in names])
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, frame=ids, q=q, hit=hit, config=json.dumps(dataclasses.asdict(cfg)))
    os.replace(tmp, out_path)
    top3 = coll = any_safe = 0
    spread, distinct = [], []
    for k in np.flatnonzero(ok):
        ev, ed = expert_action_index(frames[k][0], cfg)
        flat = q[k].reshape(-1)
        top3 += int(np.sum(flat >= q[k, ev, ed]) - 1 < 3)      # ties count against the expert, so tied columns cannot flatter the share
        coll += int(hit[k, ev, ed])
        any_safe += int(not hit[k].all())
        spread.append(float(flat.max() - flat.min()))
        distinct.append(len(np.unique(np.round(flat, 4))))
    n_ok = int(ok.sum())
    return {"route": route_dir, "skipped": False, "frames": len(frames), "labelled": n_ok, "sec": time.time() - t0,
            "expert_top3_share": top3 / max(n_ok, 1), "expert_collision_share": coll / max(n_ok, 1), "safe_action_share": any_safe / max(n_ok, 1),
            "spread": float(np.mean(spread)) if spread else 0.0, "distinct_q": float(np.mean(distinct)) if distinct else 0.0}


def _job(args):
    route_dir, tag, cfg_dict, overwrite = args
    try:
        return label_route_dir(route_dir, tag, QConfig(**cfg_dict), overwrite)
    except Exception as e:  # one broken route must not end the run
        return {"route": route_dir, "skipped": False, "error": repr(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tag", default="qcf_h12")
    ap.add_argument("--workers", type=int, default=_usable_cpus(), help="default: the CPUs this process may run on (affinity mask and cgroup "
                    "quota), not os.cpu_count(), which reports the host's threads inside a Vast container")
    ap.add_argument("--limit", type=int, default=0, help="label only the first N routes (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--horizon", type=int, default=QConfig.horizon)
    ap.add_argument("--hold", type=int, default=QConfig.hold)
    ap.add_argument("--gamma", type=float, default=QConfig.gamma)
    ap.add_argument("--summary", default=None, help="write the per-route stats here (JSON)")
    a = ap.parse_args()
    cfg = dataclasses.replace(QConfig(), horizon=a.horizon, hold=a.hold, gamma=a.gamma)
    routes = find_routes(a.data_dir)
    if a.limit:
        routes = routes[: a.limit]
    print(f"--> {len(routes)} routes, {a.workers} workers, config {dataclasses.asdict(cfg)}", flush=True)
    t0, results = time.time(), []
    with Pool(max(1, a.workers)) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, [(rd, a.tag, dataclasses.asdict(cfg), a.overwrite) for rd in routes])):
            results.append(r)
            if (i + 1) % 25 == 0 or i + 1 == len(routes):
                el = time.time() - t0
                print(f"--> {i + 1}/{len(routes)} routes, {el / 60:.1f} min, ~{el / (i + 1) * (len(routes) - i - 1) / 60:.1f} min left", flush=True)
    done = [r for r in results if not r.get("skipped") and "error" not in r]
    errors = [r for r in results if "error" in r]
    lab = sum(r["labelled"] for r in done)
    w = lambda k: sum(r[k] * r["labelled"] for r in done) / max(lab, 1)
    summary = {"routes": len(routes), "labelled_routes": len(done), "skipped": sum(bool(r.get("skipped")) for r in results), "errors": len(errors),
               "frames": sum(r["frames"] for r in done), "labelled_frames": lab, "expert_top3_share": w("expert_top3_share"),
               "expert_collision_share": w("expert_collision_share"), "safe_action_share": w("safe_action_share"), "spread": w("spread"),
               "distinct_q": w("distinct_q"), "wall_min": (time.time() - t0) / 60,
               "sec_per_frame_cpu": sum(r["sec"] for r in done) / max(sum(r["frames"] for r in done), 1)}
    print(json.dumps(summary, indent=1), flush=True)
    for e in errors[:5]:
        print("ERROR", e["route"], e["error"], flush=True)
    if a.summary:
        with open(a.summary, "w") as f:
            json.dump({"summary": summary, "config": dataclasses.asdict(cfg), "routes": results}, f, indent=1)
    try:  # MLflow (project rule); tracking must never fail the build
        import mlflow
        from src.config.paths import mlruns_dir
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:" + str(mlruns_dir())))
        mlflow.set_experiment("A71_q_labels")
        with mlflow.start_run(run_name=a.tag):
            mlflow.log_params({**{k: str(v) for k, v in dataclasses.asdict(cfg).items()}, "data_dir": a.data_dir, "tag": a.tag})
            mlflow.log_metrics({k: float(v) for k, v in summary.items()})
            if a.summary:
                mlflow.log_artifact(a.summary)
    except Exception as e:
        print("mlflow skipped:", e)
    print(f"Q_LABELS {a.tag} {lab} frames, {len(errors)} errors")
    return 1 if errors and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
