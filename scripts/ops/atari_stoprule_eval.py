"""Atari screening evaluator (scratch tool, not the paper protocol): the official evaluate() setup (no sticky actions, no FIRE reset,
27,000-step cap, noise-free graphed search, one full game per env) plus a labelled stop rule: a game with no score change for --gap steps is
ended with its current score. B50 measured the longest gap between scoring events at 65-78 steps and the last scoring point before step 1,814
on four checkpoints, so gap 2000 reproduced the official scores there; the rule makes a screen ~7-13x faster. Also takes --sims and
--max-depth (TODO B43) so search depth / simulation sweeps fit the time budget.
usage: stoprule_eval.py --ckpt X.pt --episodes 30 --sims 16 [--max-depth 5] [--gap 2000] --out result.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/MThesis")
from atari_qwen.training.eval_ez_checkpoints import load_model  # noqa: E402
from atari_qwen.training import train_ez_offpolicy as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--gap", type=int, default=2000, help="end a game after this many steps without a score change (0 = off)")
    ap.add_argument("--max-steps", type=int, default=27_000)
    ap.add_argument("--sticky", type=float, default=0.0, help="ALE sticky-action probability (0 = paper protocol; 0.25 = the separate robustness column)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck, args, model, mcts = load_model(Path(a.ckpt), dev, sims=a.sims, max_depth=a.max_depth)
    model.eval()
    graphed = T.GraphedSearch(mcts, model)
    envs = T.make_vector_atari_envs(args.env_id, num_envs=a.episodes, seed=args.seed + 10_000, clip_reward=False, episodic_life=False,
                                    frame_size=args.frame_size, grayscale=False, fire_reset=False, max_episode_steps=a.max_steps,
                                    repeat_action_probability=a.sticky)
    obs, _ = envs.reset(seed=args.seed + 10_000)
    n = a.episodes
    scores = np.zeros(n)
    finished = np.zeros(n, dtype=bool)
    stopped = np.zeros(n, dtype=bool)
    last = np.zeros(n, dtype=int)         # step of the last score change
    end_step = np.zeros(n, dtype=int)
    t0 = time.time()
    steps = 0
    for step in range(a.max_steps):
        if finished.all():
            break
        _, _, act = T.run_search(model, mcts, T.to_input(obs, dev), add_noise=False, flip_perm=None, graphed=graphed)
        obs, r, term, trunc, _ = envs.step(act)
        r = np.where(finished, 0.0, r)
        scores += r
        last[r != 0] = step + 1
        newly = (~finished) & np.logical_or(term, trunc)
        if a.gap > 0:
            stop = (~finished) & (~newly) & ((step + 1 - last) >= a.gap)
            stopped |= stop
            newly |= stop
        end_step[newly] = step + 1
        finished |= newly
        steps = step + 1
        if step % 1000 == 999:
            print(f"  step {steps}: games over {int(finished.sum())}/{n}, stopped by the rule {int(stopped.sum())}, "
                  f"{steps / (time.time() - t0):.1f} steps/s", flush=True)
    envs.close()
    wall = time.time() - t0
    res = dict(ckpt=a.ckpt, episodes=n, sims=a.sims, max_depth=a.max_depth, gap=a.gap, sticky=a.sticky, steps=steps, wall_s=wall,
               mean=float(scores.mean()), scores=scores.tolist(), stopped=int(stopped.sum()), end_step=end_step.tolist(),
               distinct_scores=int(len(set(scores.tolist()))), torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    json.dump(res, open(a.out, "w"), indent=1)
    try:  # CLAUDE.md: every eval script logs to MLflow (file store)
        import os
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
        mlflow.set_tracking_uri("file:///workspace/mlruns_screen")
        mlflow.set_experiment("B43_stoprule_screen")
        with mlflow.start_run(run_name=Path(a.out).stem):
            mlflow.log_params(dict(ckpt=Path(a.ckpt).name, episodes=n, sims=a.sims, max_depth=a.max_depth or 0, gap=a.gap, sticky=a.sticky))
            mlflow.log_metrics(dict(mean_score=res["mean"], wall_s=wall, steps=steps, stopped_by_rule=res["stopped"], distinct_scores=res["distinct_scores"]))
            mlflow.log_artifact(a.out)
    except Exception as ex:
        print("mlflow logging skipped:", ex)
    print(json.dumps({k: v for k, v in res.items() if k not in ("scores", "end_step")}), flush=True)


if __name__ == "__main__":
    main()
