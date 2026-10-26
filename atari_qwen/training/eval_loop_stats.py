"""TODO B44 step 0 (2026-10-15): how many zero scores are stuck loops?

Re-scores saved checkpoints with the same search and env settings as `evaluate()` (no sticky actions, raw reward, search without
noise, one full game per parallel env, step cap), and records per game: score, length, the longest run of steps without any
reward, and the number of FIRE presses. A zero-score game counts as a loop when it reached the step cap or its longest
reward-free streak is at least STREAK_STOP steps (the B50 stop rule). Screen-level evidence only: the paper number is not
produced here.

usage: python atari_qwen/training/eval_loop_stats.py CKPT [CKPT ...] --episodes 30 --sims 16 --max-steps 27000 --out DIR
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from atari_qwen.envs.atari_wrappers import gym, make_vector_atari_envs
from atari_qwen.mcts.gumbel_mcts import GraphedSearch
from atari_qwen.training.eval_ez_checkpoints import load_model, mlflow_run
from atari_qwen.training.train_ez_offpolicy import run_search, to_input

STREAK_STOP = 2000  # B50's stop rule: no score for 2,000 steps


def game_stats(model, mcts, env_id, device, episodes, frame_size, seed, max_steps, stagnation=0, sticky=0.0, graph=False, noise=False):
    """stagnation > 0 (TODO B44 step 1, a labelled protocol row): once a game has gone `stagnation` steps without reward,
    its actions are sampled at random until the next reward. Off (0) = the step 0 measurement.
    graph (TODO B64): replay the search from a CUDA graph as `evaluate(graph=True)` does (same results; the eager search is
    ~7,400 launches per step and bound by one CPU core).
    noise (TODO B64): Gumbel noise at the root, as the training search acts (False = the greedy screen). Also returns the mean
    probability of FIRE that the search (pi_root_fire) and the raw network prior (pi_prior_fire) give the live games."""
    graphed = GraphedSearch(mcts, model) if graph and device.type == "cuda" else None
    probe = gym.make(env_id)
    meanings = probe.unwrapped.get_action_meanings()
    probe.close()
    fire_id = meanings.index("FIRE") if "FIRE" in meanings else -1
    envs = make_vector_atari_envs(env_id, num_envs=episodes, seed=seed + 10_000, clip_reward=False, episodic_life=False,
                                  frame_size=frame_size, grayscale=False, fire_reset=False, max_episode_steps=max_steps,
                                  repeat_action_probability=sticky)
    obs, _ = envs.reset(seed=seed + 10_000)
    scores = np.zeros(episodes)
    length = np.zeros(episodes, dtype=np.int64)
    streak = np.zeros(episodes, dtype=np.int64)
    max_streak = np.zeros(episodes, dtype=np.int64)
    fires = np.zeros(episodes, dtype=np.int64)
    finished = np.zeros(episodes, dtype=bool)
    model.eval()
    rng = np.random.default_rng(seed)
    n_actions = envs.single_action_space.n
    t0 = time.time()
    pi_root_sum = pi_prior_sum = 0.0
    n_live = 0
    if noise:
        torch.manual_seed(seed + 1)
    for step in range(max_steps):
        if finished.all():
            break
        x = to_input(obs, device)
        _, pi, a = run_search(model, mcts, x, add_noise=noise, graphed=graphed)
        if fire_id >= 0:
            with torch.no_grad():
                _, _, plog = model.initial_inference(x)
            prior = torch.softmax(plog.float(), -1)[:, fire_id].cpu().numpy()
            live_now = ~finished
            pi_root_sum += float(pi[live_now, fire_id].sum())
            pi_prior_sum += float(prior[live_now].sum())
            n_live += int(live_now.sum())
        a = np.asarray(a, dtype=np.int64)
        if stagnation > 0:
            stagnant = streak >= stagnation  # no reward for `stagnation` steps: random action until the next reward
            a = np.where(stagnant, rng.integers(0, n_actions, size=episodes), a)
        obs, r, term, trunc, _ = envs.step(a)
        live = ~finished
        scores += np.where(live, r, 0.0)
        length += live
        fires += (live & (a == fire_id)).astype(np.int64)
        streak = np.where(live & (r != 0), 0, streak + live)
        max_streak = np.maximum(max_streak, streak)
        finished |= np.logical_or(term, trunc)
        if step % 2000 == 1999:
            print(f"  step {step + 1}, games over {int(finished.sum())}/{episodes}, {(time.time() - t0) / 60:.1f} min", flush=True)
    envs.close()
    if graphed is not None and graphed.fallbacks:
        print(f"  graphed search: {graphed.fallbacks} of {graphed.calls} searches exceeded the depth bound (eager fallback)", flush=True)
    return {"scores": scores, "length": length, "max_streak": max_streak, "fires": fires, "finished": finished,
            "minutes": (time.time() - t0) / 60,
            "pi_root_fire": pi_root_sum / n_live if n_live else None,
            "pi_prior_fire": pi_prior_sum / n_live if n_live else None}


def summarise(g, max_steps):
    zero = g["scores"] == 0
    capped = g["length"] >= max_steps
    loop = zero & (capped | (g["max_streak"] >= STREAK_STOP))
    n_zero = int(zero.sum())
    return {
        "episodes": int(len(zero)), "mean": float(g["scores"].mean()), "median": float(np.median(g["scores"])),
        "distinct_scores": int(len(np.unique(g["scores"]))), "zero_scores": n_zero,
        "zero_loops": int(loop.sum()), "zero_loop_share": float(loop.sum() / n_zero) if n_zero else None,
        "capped_games": int(capped.sum()), "median_length": float(np.median(g["length"])),
        "minutes": g["minutes"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", type=Path)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=27_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stagnation", type=int, default=0, help="B44 step 1 protocol row: random actions after this many reward-free steps (0 = off)")
    ap.add_argument("--sticky", type=float, default=0.0, help="B2 robustness row: ALE sticky-action probability (0.25 = B2's protocol)")
    ap.add_argument("--pess-lambda", type=float, default=None,
                    help="B38 step 0: pessimistic leaf values with the checkpoint's target network (lambda in {0, 0.5, 1}); omit = control")
    ap.add_argument("--graph", action="store_true", help="TODO B64: CUDA-graph search (same results, fewer CPU launches)")
    ap.add_argument("--noise", action="store_true",
                    help="TODO B64: Gumbel noise at the root (the training search's acting mode) instead of the greedy search")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ck_path in args.ckpts:
        ck, a, model, mcts = load_model(ck_path, device, sims=args.sims)
        if args.pess_lambda is not None:  # B38 step 0: the target network sits next to the online one in the checkpoint
            import copy
            target = copy.deepcopy(model)
            target.load_state_dict(ck["target"])
            target.eval()
            mcts.target, mcts.pess_lambda = target, args.pess_lambda
        label = (f"{ck_path.parent.parent.name}_{ck_path.stem}" + (f"_stag{args.stagnation}" if args.stagnation else "")
                 + (f"_sticky{int(args.sticky * 100):02d}" if args.sticky else "")
                 + (f"_pess{args.pess_lambda:g}_sims{args.sims}" if args.pess_lambda is not None else "")
                 + ("_noise" if args.noise else ""))
        print(f"== {label}: {args.episodes} games, {args.sims} sims, cap {args.max_steps}, stagnation {args.stagnation or 'off'}, "
              f"sticky {args.sticky:g}", flush=True)
        g = game_stats(model, mcts, a.env_id, device, args.episodes, a.frame_size, a.seed, args.max_steps, args.stagnation, args.sticky,
                       graph=args.graph, noise=args.noise)
        summ = summarise(g, args.max_steps)
        summ.update(pi_root_fire=g["pi_root_fire"], pi_prior_fire=g["pi_prior_fire"])
        out = {"checkpoint": str(ck_path), "label": label, "sims": args.sims, "max_steps": args.max_steps,
               "streak_stop": STREAK_STOP, "stagnation": args.stagnation, "sticky": args.sticky, "summary": summ,
               "per_game": [{"score": float(s), "length": int(l), "max_reward_free_streak": int(m), "fires": int(f)}
                            for s, l, m, f in zip(g["scores"], g["length"], g["max_streak"], g["fires"])]}
        (args.out / f"{label}_loops.json").write_text(json.dumps(out, indent=1))
        mlf, run = mlflow_run(label, a, proto={"episodes": args.episodes, "sims": args.sims, "sticky": 0.0, "tag": "b44_loops"},
                              tag="b44_loops")
        if mlf is not None:
            try:
                mlf.log_params({"max_steps": args.max_steps, "streak_stop": STREAK_STOP})
                mlf.log_metrics({k: float(v) for k, v in summ.items() if v is not None})
            finally:
                mlf.end_run()
        print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
