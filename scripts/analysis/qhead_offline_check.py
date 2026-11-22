"""TODO A71: offline check of a trained Q-critic head against the counterfactual Q labels on the held-out split (no simulator).

Overall metrics (rank correlation, argmax agreement, speed / lateral argmax agreement) and the same on the bypass frames: frames where the
expert bends its route off the plan (`route` vs `route_original`, the lateral index build_q_labels.expert_action_index gives). The overall lateral
agreement is dominated by in-lane frames (95.5% for the 2026-11-01 smoke head, which then stopped behind the parked car on route 2664), so the
bypass numbers are the ones that say whether the head would pass an obstacle:
  * bypass_label_expert_side_share - the label's best action leaves the lane on the expert's side (a property of the labels);
  * bypass_pred_expert_side_share  - the head's best action does (a property of the head);
  * decode_leave_inlane_m<m> / decode_expert_side_bypass_m<m> - the decode rule at lat_margin m (0.5 -> m0p5): lane changes on in-lane frames (noise) and the right
    side on bypass frames; pick the closed-loop lat_margin from these.

    python scripts/analysis/qhead_offline_check.py --ckpt /workspace/checkpoints/A71_q1/model_epoch_020.pth --out /workspace/a71/qhead_check.json
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts", "training"))

from src.training.counterfactual_q import QConfig  # noqa: E402


def _rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def takes_side(q: np.ndarray, side: int, eps: float = 1e-3) -> bool:
    """The best column on `side` beats the own lane by more than eps (np.argmax over tied columns would pick the leftmost: a left bias)."""
    lat = np.asarray(QConfig().lat_targets)
    v = q.max(0)
    return bool(v[np.sign(lat) == side].max() > v[lat == 0.0].max() + eps)


def decoded_offset(q: np.ndarray, margin: float) -> float:
    """The lane the decode rule (q_critic.decode_with_q, from the own lane) picks: Q only, so no BC is needed offline."""
    from src.models.world_on_rails.q_critic import decode_with_q
    from src.models.world_on_rails.aux_heads import TARGET_SPEEDS
    n = len(TARGET_SPEEDS)
    return decode_with_q(np.full(n, 1.0 / n), q, np.asarray(TARGET_SPEEDS), beta=1.0, lat_margin=margin)[1]


def qhead_metrics(pred: np.ndarray, tgt: np.ndarray, side: np.ndarray, margins=(0.5, 0.75, 1.0, 1.5, 2.0)) -> dict:
    """pred, tgt (N, n_speed, n_lat); side (N,) the expert's lateral side (-1 left, 0 in lane, +1 right). Also the decode rule's calibration over
    lat_margin: decode_leave_inlane@m (in-lane frames where the rule leaves the lane: noise) and decode_expert_side_bypass@m."""
    lat = np.asarray(QConfig().lat_targets)
    n = len(pred)
    rho, top1, top3, v_ok, d_ok, err, lab_side, pred_side = [], 0, 0, 0, 0, [], [], []
    for p, t, s in zip(pred, tgt, side):
        pf, tf = p.reshape(-1), t.reshape(-1)
        if pf.std() > 1e-6 and tf.std() > 1e-6:
            rho.append(float(np.corrcoef(_rank(pf), _rank(tf))[0, 1]))
        b = int(np.argmax(pf))
        top1 += int(tf[b] >= tf.max() - 1e-5)
        top3 += int(np.sum(tf > tf[b] + 1e-5) < 3)
        v_ok += int(np.argmax(p.max(axis=1)) == np.argmax(t.max(axis=1)))
        d_ok += int(np.argmax(p.max(axis=0)) == np.argmax(t.max(axis=0)))
        err.append(float(np.abs(pf - tf).mean()))
        if s != 0:
            lab_side.append(takes_side(t, s))
            pred_side.append(takes_side(p, s))
    share = lambda xs: float(np.mean(xs)) if xs else None
    calib = {}
    for m in margins:
        dec = np.array([decoded_offset(p, m) for p in pred])
        tag = f"m{m:g}".replace(".", "p")                                     # MLflow rejects '@' in metric names
        calib[f"decode_leave_inlane_{tag}"] = share(list(dec[side == 0] != 0.0))
        calib[f"decode_expert_side_bypass_{tag}"] = share(list(np.sign(dec[side != 0]) == side[side != 0]))
    return {"frames": n, "mean_abs_error": float(np.mean(err)) if err else None,
            "mean_spearman_rho": share(rho), "median_spearman_rho": float(np.median(rho)) if rho else None,
            "argmax_in_label_top1_share": top1 / max(n, 1), "argmax_in_label_top3_share": top3 / max(n, 1),
            "speed_argmax_agreement": v_ok / max(n, 1), "lat_argmax_agreement": d_ok / max(n, 1),
            "bypass_frames": len(lab_side), "bypass_label_expert_side_share": share(lab_side), "bypass_pred_expert_side_share": share(pred_side),
            **calib}


def expert_side(meas: dict) -> int:
    from build_q_labels import expert_action_index
    cfg = QConfig()
    _, d = expert_action_index(meas, cfg)
    return int(np.sign(cfg.lat_targets[d]))


def measurement_path(rgb_path: str) -> str:
    route_dir, name = os.path.split(os.path.dirname(rgb_path))[0], os.path.basename(rgb_path)
    return os.path.join(route_dir, "measurements", name.split(".")[0] + ".json.gz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run_config", default=None, help="default: run_config.json next to the checkpoint")
    ap.add_argument("--q_labels", default=None, help="default: the run's q_labels")
    ap.add_argument("--data_dir", default=None, help="default: the run's data_dir")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from src.models.world_on_rails.wor_loader import load_wor_model
    from src.training.wor_dataloaders import create_wor_train_val_dataloaders
    from src.training.wor_eval import to_device_batch_with_features

    with open(a.run_config or os.path.join(os.path.dirname(a.ckpt), "run_config.json")) as f:
        rc = json.load(f)
    os.environ["WOR_ROUTE_KEY"] = rc.get("route_key", "route")
    os.environ["WOR_Q_LABELS"] = a.q_labels or rc["q_labels"]
    os.environ.setdefault("WOR_Q_MIN_COVERAGE", "0.2")                        # strided labels (build_q_labels --stride 4) cover ~25-50%
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_wor_model(checkpoint_path=a.ckpt, backbone_name=rc["backbone"], pretrained_backbone=bool(rc["pretrained"]),
                           freeze_backbone=bool(rc["freeze_backbone"]), device=dev, policy_arch=rc["policy_arch"], route_points=rc["route_points"])
    model.eval()
    h, w = (int(x) for x in str(rc["img_size"]).lower().split("x"))
    _, val_loader = create_wor_train_val_dataloaders(
        data_dir=a.data_dir or rc["data_dir"], val_data_dir=rc.get("val_data_dir"), val_split=rc.get("val_split", 0.15),
        split_seed=rc.get("split_seed", 0), fold=rc.get("fold", 0), num_folds=rc.get("num_folds", 1), seed=rc.get("seed", 0),
        batch_size=a.batch_size, num_workers=a.num_workers, cache_decoded=False, route_points=rc["route_points"], img_size=(h, w),
        crop_bottom_frac=rc.get("crop_bottom_frac", 0.0), route_overlay=bool(rc.get("route_overlay", 0)),
        feature_cache_tag=rc.get("feature_cache_tag"), use_augmented_camera=False, color_aug_prob=0.0)
    if val_loader is None:
        raise SystemExit("no validation split (val_split 0 and no val_data_dir): nothing to check")
    vd = val_loader.dataset                                                    # shuffle off: batches follow idx in order
    ds, idx = (vd.dataset, vd.indices) if hasattr(vd, "indices") else (vd, range(len(vd)))

    preds, tgts, sides, k = [], [], [], 0
    with torch.no_grad():
        for batch in val_loader:
            rgb, vf, speed, command, route, _, _ = to_device_batch_with_features(batch, dev)
            q = model(rgb, speed, command, route, vision_features=vf)["q_sa"].float().cpu().numpy()
            valid = batch["qsa_valid"].numpy() > 0.5
            tq = batch["target_qsa"].float().numpy().reshape(q.shape)
            for i in range(len(q)):
                j = idx[k + i]
                if valid[i]:
                    with gzip.open(measurement_path(ds.samples[j]["rgb_path"]), "rt") as f:
                        sides.append(expert_side(json.load(f)))
                    preds.append(q[i])
                    tgts.append(tq[i])
            k += len(q)
    if not preds:
        raise SystemExit(f"no labelled validation frame under q_labels={os.environ['WOR_Q_LABELS']}")
    res = {"ckpt": a.ckpt, "q_labels": os.environ["WOR_Q_LABELS"], **qhead_metrics(np.stack(preds), np.stack(tgts), np.array(sides))}
    print("QHEAD_CHECK", json.dumps(res, indent=1), flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    try:  # MLflow (project rule); tracking must never fail the check
        import mlflow
        from src.config.paths import mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri())
        mlflow.set_experiment("A71_qhead_check")
        with mlflow.start_run(run_name=os.path.basename(os.path.dirname(a.ckpt)) + "/" + os.path.basename(a.ckpt)):
            mlflow.log_params({"ckpt": a.ckpt, "q_labels": res["q_labels"]})
            mlflow.log_metrics({k: float(v) for k, v in res.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
            mlflow.log_artifact(a.out)
    except Exception as e:
        print("mlflow skipped:", e)


if __name__ == "__main__":
    main()
