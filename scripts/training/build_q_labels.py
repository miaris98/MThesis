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
  * bypass_argmax_expert_side_share - on frames where the expert bends its route off the plan, how often the best label action leaves the
    lane on the same side; low values mean the labels would teach the policy to stop behind an obstacle instead of passing it.

    python scripts/training/build_q_labels.py --data_dir /workspace/dataset/wor_trajectories --tag qcf_h12 [--workers 32] [--limit 5]
"""
import argparse
import dataclasses
import gzip
import json
import re
import os
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # one process per route: BLAS threads would oversubscribe the cores
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.training.counterfactual_q import logged_step_reward, QConfig, label_route  # noqa: E402


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


def scenario_of(route_dir: str) -> str:
    """PDM-Lite layout <town>/data/<Scenario>/<RouteN_RepM>: the scenario is the route folder's parent."""
    return os.path.basename(os.path.dirname(os.path.normpath(route_dir)))


def apply_overrides(cfg: QConfig, sets) -> QConfig:
    """--set key=value for any QConfig field (ablations such as hold=8 or overspeed_margin=0), cast to the field's type."""
    fields = {f.name: f for f in dataclasses.fields(QConfig)}
    kw = {}
    for item in sets or []:
        k, _, v = item.partition("=")
        if k not in fields or isinstance(getattr(cfg, k), tuple):
            raise ValueError(f"--set {item}: not a scalar QConfig field ({', '.join(n for n in fields if not isinstance(getattr(cfg, n), tuple))})")
        kw[k] = type(getattr(cfg, k))(float(v)) if isinstance(getattr(cfg, k), (int, float)) else v
    return dataclasses.replace(cfg, **kw)


def stride_for(route_dir: str, stride: int, dense_regex: str, dense_stride: int) -> int:
    return dense_stride if dense_regex and re.search(dense_regex, scenario_of(route_dir)) else stride


def label_route_dir(route_dir: str, tag: str, cfg: QConfig, overwrite: bool = False, stride: int = 1, dense_regex: str = "",
                    dense_stride: int = 1) -> dict:
    out_path = os.path.join(route_dir, f"{tag}.npz")
    if os.path.exists(out_path) and not overwrite:
        with np.load(out_path, allow_pickle=False) as z:
            if "config" in z.files and json.loads(str(z["config"])) != json.loads(json.dumps(dataclasses.asdict(cfg))):
                raise ValueError(f"{out_path} was built with another config: pass --overwrite or use a new --tag")
            if (int(z["stride"]) if "stride" in z.files else 1) != stride_for(route_dir, stride, dense_regex, dense_stride):
                raise ValueError(f"{out_path} was built with another frame stride: pass --overwrite or use a new --tag")
        return {"route": route_dir, "skipped": True}
    t0 = time.time()
    names = sorted(n for n in os.listdir(os.path.join(route_dir, "measurements")) if n.endswith(".json.gz"))
    frames = []
    for n in names:
        m, b = _read(os.path.join(route_dir, "measurements", n)), _read(os.path.join(route_dir, "boxes", n))
        frames.append((m, b) if m is not None and b is not None and "ego_matrix" in m else None)
    st = stride_for(route_dir, stride, dense_regex, dense_stride)
    q, ok, hit = label_route(frames, cfg, st)
    ids = np.array([n.split(".")[0] for n in names])
    tmp = out_path + ".tmp.npz"
    # TODO A72: the expert's action and the reward of its logged step, for every frame with a successor (not only the strided Q frames)
    a_exp = np.full(len(frames), -1, dtype=np.int16)
    r_exp = np.full(len(frames), np.nan, dtype=np.float32)
    for k in range(len(frames) - 1):
        if frames[k] is not None and frames[k + 1] is not None:
            ev, ed = expert_action_index(frames[k][0], cfg)
            a_exp[k], r_exp[k] = ev * len(cfg.lat_targets) + ed, logged_step_reward(frames, k, cfg)
    np.savez_compressed(tmp, frame=ids, q=q, hit=hit, config=json.dumps(dataclasses.asdict(cfg)), stride=st, a_expert=a_exp, r_expert=r_exp)
    os.replace(tmp, out_path)
    top3 = coll = any_safe = n_bypass = bypass_side = 0
    spread, distinct = [], []
    lat = np.asarray(cfg.lat_targets)
    for k in np.flatnonzero(ok):
        ev, ed = expert_action_index(frames[k][0], cfg)
        flat = q[k].reshape(-1)
        if lat[ed] != 0.0:                                     # the expert is bypassing: does the best column on its side beat the own lane?
            n_bypass += 1                                      # (not np.argmax: it takes the leftmost of tied columns, a left bias)
            v_col = q[k].max(0)
            bypass_side += int(v_col[np.sign(lat) == np.sign(lat[ed])].max() > v_col[lat == 0.0].max() + 1e-3)
        top3 += int(np.sum(flat >= q[k, ev, ed]) - 1 < 3)      # ties count against the expert, so tied columns cannot flatter the share
        coll += int(hit[k, ev, ed])
        any_safe += int(not hit[k].all())
        spread.append(float(flat.max() - flat.min()))
        distinct.append(len(np.unique(np.round(flat, 4))))
    n_ok = int(ok.sum())
    return {"route": route_dir, "skipped": False, "frames": len(frames), "labelled": n_ok, "sec": time.time() - t0,
            "expert_top3_share": top3 / max(n_ok, 1), "expert_collision_share": coll / max(n_ok, 1), "safe_action_share": any_safe / max(n_ok, 1),
            "spread": float(np.mean(spread)) if spread else 0.0, "distinct_q": float(np.mean(distinct)) if distinct else 0.0,
            "bypass_frames": n_bypass, "bypass_argmax_expert_side_share": bypass_side / n_bypass if n_bypass else None}


def _job(args):
    route_dir, tag, cfg_dict, overwrite, stride, dense_regex, dense_stride = args
    try:
        return label_route_dir(route_dir, tag, QConfig(**cfg_dict), overwrite, stride, dense_regex, dense_stride)
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
    ap.add_argument("--stride", type=int, default=1, help="label every N-th frame (4 Hz frames are near-duplicates; the rest carry no label)")
    ap.add_argument("--dense_regex", default="", help="scenarios matching this regex use --dense_stride instead (obstacle scenarios)")
    ap.add_argument("--dense_stride", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any scalar QConfig field (repeatable)")
    ap.add_argument("--summary", default=None, help="write the per-route stats here (JSON)")
    a = ap.parse_args()
    cfg = apply_overrides(dataclasses.replace(QConfig(), horizon=a.horizon, hold=a.hold, gamma=a.gamma), a.set)
    routes = find_routes(a.data_dir)
    if a.limit:
        routes = routes[: a.limit]
    print(f"--> {len(routes)} routes, {a.workers} workers, config {dataclasses.asdict(cfg)}", flush=True)
    t0, results = time.time(), []
    with Pool(max(1, a.workers)) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, [(rd, a.tag, dataclasses.asdict(cfg), a.overwrite, a.stride, a.dense_regex, a.dense_stride) for rd in routes])):
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
               "sec_per_frame_cpu": sum(r["sec"] for r in done) / max(lab, 1)}
    nb = sum(r["bypass_frames"] for r in done)
    summary["bypass_frames"] = nb
    summary["bypass_argmax_expert_side_share"] = (sum((r["bypass_argmax_expert_side_share"] or 0.0) * r["bypass_frames"] for r in done) / nb
                                                  if nb else -1.0)
    per_scenario = {}
    for sc in sorted({scenario_of(r["route"]) for r in done}):
        rs = [r for r in done if scenario_of(r["route"]) == sc]
        n, b = sum(r["labelled"] for r in rs), sum(r["bypass_frames"] for r in rs)
        per_scenario[sc] = {"routes": len(rs), "labelled_frames": n, "bypass_frames": b,
                            **{k: sum(r[k] * r["labelled"] for r in rs) / max(n, 1) for k in ("expert_top3_share", "expert_collision_share", "safe_action_share")},
                            "bypass_argmax_expert_side_share": sum((r["bypass_argmax_expert_side_share"] or 0.0) * r["bypass_frames"] for r in rs) / b if b else None}
    print(json.dumps(summary, indent=1), flush=True)
    print(json.dumps(per_scenario, indent=1), flush=True)
    for e in errors[:5]:
        print("ERROR", e["route"], e["error"], flush=True)
    if a.summary:
        with open(a.summary, "w") as f:
            json.dump({"summary": summary, "per_scenario": per_scenario, "config": dataclasses.asdict(cfg), "routes": results}, f, indent=1)
    try:  # MLflow (project rule); tracking must never fail the build
        import mlflow
        from src.config.paths import mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri())
        mlflow.set_experiment("A71_q_labels")
        with mlflow.start_run(run_name=a.tag):
            mlflow.log_params({**{k: str(v) for k, v in dataclasses.asdict(cfg).items()}, "data_dir": a.data_dir, "tag": a.tag, "stride": a.stride,
                               "dense_regex": a.dense_regex, "dense_stride": a.dense_stride})
            mlflow.log_metrics({k: float(v) for k, v in summary.items()})
            if a.summary:
                mlflow.log_artifact(a.summary)
    except Exception as e:
        print("mlflow skipped:", e)
    print(f"Q_LABELS {a.tag} {lab} frames, {len(errors)} errors")
    return 1 if errors and not done else 0


if __name__ == "__main__":
    raise SystemExit(main())
