"""TODO A73: offline gate of a trained latent world model, before any CARLA run (TODO_ACTIVE A73 step 5).

On held-out route windows:
  * latent_err_k / copy_err_k  - k-step latent prediction error against the trivial "nothing changes" prediction (z_0 for every k); < 1 means
                                 the dynamics learnt something, ~1 means a collapsed or copying model;
  * disagreement_ratio         - ensemble spread on counterfactual actions / on the expert's actions; it must rise (> 1) where data is absent,
                                 or the penalty cannot keep the policy on the data;
  * reward_corr, reward_std_ratio, reward_collapsed - the reward head against the logged step rewards.

    python scripts/analysis/a73_world_model_gate.py --ckpt /workspace/checkpoints/A73_q/wm_epoch_010.pth --out /workspace/a73/gate.json
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)


def latent_error_vs_copy(model_err, copy_err):
    """Per-step ratio of the model's latent error to the copy-z_0 baseline's."""
    return list(np.asarray(model_err, dtype=np.float64) / np.maximum(np.asarray(copy_err, dtype=np.float64), 1e-12))


def disagreement_ratio(u_expert, u_counterfactual) -> float:
    return float(np.mean(u_counterfactual) / max(float(np.mean(u_expert)), 1e-12))


def reward_collapse(pred, target, min_std_ratio: float = 0.1, min_corr: float = 0.1) -> dict:
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    sr = float(pred.std() / max(target.std(), 1e-12))
    corr = float(np.corrcoef(pred, target)[0, 1]) if pred.std() > 1e-12 and target.std() > 1e-12 else 0.0
    return {"corr": corr, "std_ratio": sr, "collapsed": bool(sr < min_std_ratio or corr < min_corr)}


def main(argv=None):
    import torch
    from torch.utils.data import DataLoader
    from src.models.world_on_rails.latent_world_model import LatentWorldModel, WorldModelConfig
    from src.training.latent_world_model_train import LatentWindowDataset
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_windows", type=int, default=4000)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    a = ap.parse_args(argv)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    rc = ck["run_config"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = LatentWorldModel(WorldModelConfig(**ck["model_config"])).to(dev).eval()
    model.load_state_dict(ck["model"])
    target = LatentWorldModel(WorldModelConfig(**ck["model_config"])).to(dev).eval()
    target.load_state_dict(ck["target"])
    ds = LatentWindowDataset(rc["data_dir"], rc["pooled_cache"].split(","), rc["q_labels"], K=rc["K"], routes=set(ck["val_routes"]))
    if not len(ds):
        raise SystemExit("no held-out windows in the checkpoint's val_routes (train with --val_frac > 0)")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=a.num_workers)
    K = rc["K"]
    lat, cpy, u_exp, u_cf, r_pred, r_true, n = np.zeros(K), np.zeros(K), [], [], [], [], 0
    with torch.no_grad():
        for b in loader:
            f, acts, rew = b["feats"].to(dev), b["actions"].to(dev), b["rewards"].to(dev)
            z = [target.encode(f[:, k]) for k in range(K + 1)]
            s = model.encode(f[:, 0])
            for k in range(K):
                s_mean, s_all, r = model.step(s, acts[:, k])
                lat[k] += float(((s_mean - z[k + 1]) ** 2).mean((1, 2)).sum())
                cpy[k] += float(((z[0] - z[k + 1]) ** 2).mean((1, 2)).sum())
                if k == 0:
                    u_exp += s_all.std(0).mean((1, 2)).tolist()
                    cf = (acts[:, 0] + torch.randint(1, model.cfg.n_actions, acts[:, 0].shape, device=dev)) % model.cfg.n_actions
                    u_cf += model.step(s, cf)[1].std(0).mean((1, 2)).tolist()
                    r_pred += r.tolist()
                    r_true += rew[:, 0].tolist()
                s = s_mean
            n += f.shape[0]
            if n >= a.max_windows:
                break
    res = {"ckpt": a.ckpt, "windows": n, "latent_vs_copy": latent_error_vs_copy(lat / n, cpy / n),
           "disagreement_ratio": disagreement_ratio(u_exp, u_cf), **{f"reward_{k}": v for k, v in reward_collapse(r_pred, r_true).items()}}
    res["gate_pass"] = bool(res["latent_vs_copy"][-1] < 1.0 and res["disagreement_ratio"] > 1.0 and not res["reward_collapsed"])
    print("A73_GATE", json.dumps(res, indent=1), flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    try:  # MLflow (project rule); tracking must never fail the gate
        import mlflow
        from src.config.paths import mlruns_dir
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:" + str(mlruns_dir())))
        mlflow.set_experiment("A73_world_model_gate")
        with mlflow.start_run(run_name=os.path.basename(a.ckpt)):
            mlflow.log_metrics({"disagreement_ratio": res["disagreement_ratio"], "reward_corr": res["reward_corr"],
                                "reward_std_ratio": res["reward_std_ratio"], "gate_pass": float(res["gate_pass"]),
                                **{f"latent_vs_copy_k{k + 1}": v for k, v in enumerate(res["latent_vs_copy"])}})
            mlflow.log_artifact(a.out)
    except Exception as e:
        print("mlflow skipped:", e)
    return res


if __name__ == "__main__":
    main()
