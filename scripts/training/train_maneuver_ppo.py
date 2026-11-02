"""TODO A74 (2026-11-02): the PPO learner for the maneuver head (src/models/world_on_rails/maneuver_rl.py, src/training/maneuver_ppo.py).

Runs on the RL box next to the CARLA lanes. The lanes run the Bench2Drive evaluator on training routes with
WOR_RL="head=<out>/head_latest.pth,mode=train,log=<rollouts>"; every agent reloads head_latest.pth at its route start and writes one npz per
route. This loop waits until --batch_decisions decisions are in, sets aside episodes from a head more than --max_lag updates old, runs PPO,
publishes the new head atomically (head_latest.pth + head_iterNNNN.pth) and moves the consumed episodes to <rollouts>/consumed/iterNNNN/.
Resumable: the head, the Adam state and the iteration live in <out>. MLflow experiment A74_maneuver_ppo plus <out>/learner_log.csv.

    python scripts/training/train_maneuver_ppo.py --rollouts /workspace/a74/rollouts --out /workspace/a74/learner --embed_dim 256
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

from src.models.world_on_rails.maneuver_rl import DEFAULT_ACTION, ManeuverHead, load_head, save_head  # noqa: E402
from src.training.maneuver_ppo import PPOConfig, batch_from_episodes, load_episodes, ppo_update, prior_log_probs  # noqa: E402


def init_head(out: str, embed_dim: int, margin: float = 7.0, seed: int = 0) -> ManeuverHead:
    """The head to train: <out>/head_latest.pth when resuming, else a fresh one at E15, published as iteration 0 so the lanes start from it."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "head_latest.pth")
    if os.path.exists(path):
        return load_head(path, embed_dim, margin)
    torch.manual_seed(seed)
    head = ManeuverHead(embed_dim).reset_to_default(margin)
    head.meta = {"iter": 0}
    save_head(head, path, head.meta)
    save_head(head, os.path.join(out, "head_iter0000.pth"), head.meta)
    return head.eval()


def learner_iteration(head: ManeuverHead, rollouts: str, out: str, it: int, min_decisions: int, cfg: PPOConfig, max_lag: int = 3,
                      reward_scale: float = 0.01, optimizer=None, normalize_adv: bool = True):
    """One PPO update from the finished episodes, or None while fewer than min_decisions are waiting. `it` is the update being made; the
    published head is it - 1, so an episode's lag is (it - 1) - its head_iter."""
    eps = load_episodes(rollouts)
    fresh, stale = [], []
    for ep in eps:
        (stale if (it - 1) - int(ep["info"].get("head_iter", 0)) > max_lag else fresh).append(ep)
    if stale:
        os.makedirs(os.path.join(rollouts, "stale"), exist_ok=True)
        for ep in stale:
            shutil.move(ep["path"], os.path.join(rollouts, "stale", os.path.basename(ep["path"])))
    n = int(sum(len(ep["action"]) for ep in fresh))
    if n < min_decisions or not fresh:
        return None
    for ep in fresh:
        ep["reward"] = ep["reward"].astype(np.float64) * reward_scale
    feats, acts, old_logp, adv, ret = batch_from_episodes(fresh, cfg)
    if normalize_adv and len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = ppo_update(head, feats, acts, old_logp, adv, ret, prior_log_probs(head.margin), cfg, optimizer=optimizer)
    head.meta = {"iter": it, "time": time.time()}
    save_head(head, os.path.join(out, f"head_iter{it:04d}.pth"), head.meta)
    save_head(head, os.path.join(out, "head_latest.pth"), head.meta)
    dst = os.path.join(rollouts, "consumed", f"iter{it:04d}")
    os.makedirs(dst, exist_ok=True)
    for ep in fresh:
        shutil.move(ep["path"], os.path.join(dst, os.path.basename(ep["path"])))
    infos = [ep["info"] for ep in fresh]
    stats.update({"iter": it, "episodes": len(fresh), "decisions": n, "stale_episodes": len(stale),
                  "deviation_rate": float(np.mean(np.concatenate([ep["action"] for ep in fresh]) != DEFAULT_ACTION)),
                  "return_mean": float(np.mean([i.get("return", 0.0) for i in infos])),
                  "rc_mean": float(np.mean([i.get("rc", 0.0) for i in infos])),
                  "collisions_per_route": float(np.mean([sum((i.get("collisions") or {}).values()) for i in infos]))})
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed_dim", type=int, default=0,
                    help="the policy token's width (QwenWorldOnRailsPolicy.embed_dim); 0 = read it from the first episode (the agents start "
                         "from a fresh head at E15 until the learner has published one)")
    ap.add_argument("--batch_decisions", type=int, default=4096)
    ap.add_argument("--max_decisions", type=int, default=300_000)
    ap.add_argument("--max_lag", type=int, default=3)
    ap.add_argument("--margin", type=float, default=7.0)
    ap.add_argument("--reward_scale", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--kl_coef", type=float, default=0.02)
    ap.add_argument("--ent_coef", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poll_s", type=float, default=20.0)
    ap.add_argument("--use_mlflow", type=int, default=1)
    a = ap.parse_args()

    cfg = PPOConfig(gamma=a.gamma, lam=a.lam, epochs=a.epochs, minibatch=a.minibatch, lr=a.lr, kl_coef=a.kl_coef, ent_coef=a.ent_coef)
    os.makedirs(a.rollouts, exist_ok=True)
    embed_dim = a.embed_dim
    while embed_dim <= 0:
        if os.path.exists(os.path.join(a.out, "head_latest.pth")):
            embed_dim = int(torch.load(os.path.join(a.out, "head_latest.pth"), map_location="cpu", weights_only=False)["in_dim"])
        else:
            eps = load_episodes(a.rollouts)
            if eps:
                embed_dim = int(eps[0]["feat"].shape[1])
            else:
                time.sleep(a.poll_s)
    print(f"policy token width {embed_dim}", flush=True)
    head = init_head(a.out, embed_dim, a.margin, a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head.to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    state_path = os.path.join(a.out, "learner_state.pth")
    it, total = 1, 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev, weights_only=False)
        opt.load_state_dict(st["opt"])
        it, total = int(st["iter"]) + 1, int(st["decisions"])
        print(f"resumed at update {it}, {total} decisions so far", flush=True)

    ml = None
    if a.use_mlflow:
        try:
            import mlflow
            from src.config.paths import mlruns_dir
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:" + str(mlruns_dir())))
            mlflow.set_experiment("A74_maneuver_ppo")
            mlflow.start_run(run_name=os.path.basename(os.path.normpath(a.out)))
            mlflow.log_params({k: str(v) for k, v in vars(a).items()})
            ml = mlflow
        except Exception as e:
            print("mlflow skipped:", e, flush=True)

    log_path = os.path.join(a.out, "learner_log.csv")
    t0 = time.time()
    while total < a.max_decisions:
        stats = learner_iteration(head, a.rollouts, a.out, it, a.batch_decisions, cfg, a.max_lag, a.reward_scale, optimizer=opt)
        if stats is None:
            time.sleep(a.poll_s)
            continue
        total += stats["decisions"]
        stats.update({"decisions_total": total, "wall_h": (time.time() - t0) / 3600})
        torch.save({"opt": opt.state_dict(), "iter": it, "decisions": total}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(stats))
            if new:
                w.writeheader()
            w.writerow(stats)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}), flush=True)
        if ml is not None:
            ml.log_metrics({k: float(v) for k, v in stats.items() if isinstance(v, (int, float))}, step=it)
        it += 1
    print(f"A74_LEARNER_DONE {total} decisions, {it - 1} updates", flush=True)
    if ml is not None:
        ml.log_artifact(log_path)
        ml.end_run()


if __name__ == "__main__":
    main()
