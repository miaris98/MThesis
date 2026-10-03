"""Deferred evaluation for `train_ez_offpolicy.py --eval-mode deferred` (S-111).

The trainer saves `checkpoint_env{N}_upd{U}.pt` at every eval point and keeps training; this script scores those
checkpoints in its own process with the trainer's own `evaluate()` (same seeds, same search without noise, same
27k-step cap), so the scores are the ones an inline eval would have produced. Per checkpoint it writes
`eval_env{N}_upd{U}.json`; per run it keeps `checkpoint_best.pt` (best mean so far, with the scores attached) and
`eval_summary.json`, and logs `eval/score`, `eval/se`, `eval/hns` to MLflow (run `<label>_eval`, step = env steps).

usage:
  eval_ez_checkpoints.py RUN_DIR [RUN_DIR ...]            # score whatever is there, then exit
  eval_ez_checkpoints.py RUN_DIR --watch                  # keep scoring new checkpoints until the final one is done
  eval_ez_checkpoints.py --verify CKPT                    # re-score a checkpoint that already holds inline
                                                          # eval_scores and compare (bit-exact check of the setup)
  eval_ez_checkpoints.py RUN_DIR... --episodes 30 --sims 64 [--sticky 0.25] [--flip-avg] [--env-steps 30000]
                                                          # another protocol (B2 / B24): own files, see below

Protocol overrides (TODO B2, B24, B26 step 0): --episodes / --sims / --sticky / --flip-avg re-score saved checkpoints under another evaluation
protocol. Those results go to `eval_<tag>_env{N}_upd{U}.json` and `eval_summary_<tag>.json` (tag = --protocol, by
default e.g. `ep30_sim64_st0`), add the median and the tunnel rate P(score >= 200), and never touch
`checkpoint_best.pt`: model selection stays on the training protocol. Episode i uses the same env seed as episode i of
the default eval, so without sticky actions and at the training's simulation count the first 10 of 30 episodes
should reproduce the stored 10-episode scores (a check of the setup; batch size can change GPU numerics).
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atari_qwen.training.train_ez_offpolicy import (  # noqa: E402  (same code path as an inline eval)
    EZV2Model, DiscreteSupport, GumbelMCTS, compute_hns, evaluate, make_vector_atari_envs)

CKPT_RE = re.compile(r"checkpoint_env(\d+)_upd(\d+)\.pt$")


def load_model(ck_path: Path, device, sims=None):
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    probe = make_vector_atari_envs(a.env_id, num_envs=1, seed=a.seed, clip_reward=True, episodic_life=True,
                                   frame_size=a.frame_size, grayscale=False, fire_reset=False,
                                   max_episode_steps=a.train_episode_steps, same_step_autoreset=True)
    A = probe.single_action_space.n
    probe.close()
    support = DiscreteSupport()
    model = EZV2Model(A, obs_channels=3 * 4, support=support, trunk=a.trunk, norm=a.norm,
                      state_hw=int(np.ceil(a.frame_size / 16))).to(device)
    model.load_state_dict(ck["model"])
    mcts = GumbelMCTS(A, support, num_simulations=sims or a.num_simulations, discount=a.discount ** 4,
                      lstm_horizon=a.lstm_horizon)
    return ck, a, model, mcts


def total_updates(a) -> int:
    full_online = int(a.schedule_steps * a.replay_ratio)
    online = int(a.total_steps * a.replay_ratio)
    return online + (int(full_online * a.offline_frac) if a.total_steps >= a.schedule_steps else 0)


def score(ck_path: Path, device, proto=None):
    ck, a, model, mcts = load_model(ck_path, device, proto and proto["sims"])
    t0 = time.time()
    if proto is None:
        sc = evaluate(model, mcts, a.env_id, device, a.eval_episodes, a.frame_size, a.seed)
    else:
        sc = evaluate(model, mcts, a.env_id, device, proto["episodes"] or a.eval_episodes, a.frame_size, a.seed,
                      sticky=proto["sticky"], flip_avg=proto["flip"])
    return ck, a, sc, time.time() - t0


def protocol_tag(proto, a) -> str:
    return proto["tag"] or (f"ep{proto['episodes'] or a.eval_episodes}_sim{proto['sims'] or a.num_simulations}"
                            f"_st{proto['sticky']:g}" + ("_flip" if proto["flip"] else ""))


def mlflow_run(label: str, a, proto=None, tag=None):
    try:
        import mlflow
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        try:
            from src.config.paths import mlruns_dir
            mlflow.set_tracking_uri(mlruns_dir().resolve().as_uri())
        except Exception:
            pass
        mlflow.set_experiment(a.experiment)
        run = mlflow.start_run(run_name=f"{label}_eval" + (f"_{tag}" if tag else ""))
        mlflow.log_params({"run_label": label, "trunk": a.trunk, "seed": a.seed, "env_id": a.env_id,
                           "eval_episodes": (proto and proto["episodes"]) or a.eval_episodes, "eval_mode": "deferred",
                           "eval_simulations": (proto and proto["sims"]) or a.num_simulations,
                           "eval_sticky": proto["sticky"] if proto else 0.0, "eval_protocol": tag or "train"})
        return mlflow, run
    except Exception as ex:  # MLflow is optional for the scores themselves
        print(f"! MLflow disabled ({ex})", flush=True)
        return None, None


def process_run(run_dir: Path, device, state: dict, proto=None, env_steps=None) -> bool:
    """Score every unscored checkpoint of one run. Returns True once the run's final checkpoint is scored.
    proto: protocol overrides (own result files, no checkpoint_best.pt); env_steps: only these checkpoints."""
    ckdir = run_dir / "checkpoints"
    found = []
    for p in ckdir.glob("checkpoint_env*_upd*.pt"):
        m = CKPT_RE.search(p.name)
        if m:
            found.append((int(m.group(2)), int(m.group(1)), p))
    if env_steps:
        found = [f for f in found if f[1] in env_steps]
    final_done = False
    tag = None
    for upd, env, p in sorted(found):
        if proto is not None:
            tag = protocol_tag(proto, argparse.Namespace(**torch.load(p, map_location="cpu", weights_only=False)["args"]))
        out = ckdir / (f"eval_{tag}_env{env}_upd{upd}.json" if tag else f"eval_env{env}_upd{upd}.json")
        if out.exists():
            res = json.loads(out.read_text())
            final_done |= res.get("tag") == "final"
            continue
        ck, a, sc, secs = score(p, device, proto)
        mean, se = float(sc.mean()), float(sc.std() / np.sqrt(len(sc)))
        hns = compute_hns(mean, a.env_id)
        res = {"tag": ck.get("eval_tag") or ("final" if upd >= total_updates(a) else f"env{env}"),
               "env_steps": env, "updates": upd, "mean": mean, "se": se, "hns": hns,
               "scores": sc.tolist(), "eval_seconds": round(secs, 1), "checkpoint": p.name}
        if tag:
            res.update(protocol=tag, episodes=len(sc), simulations=proto["sims"] or a.num_simulations,
                       sticky=proto["sticky"], flip_avg=proto["flip"], median=float(np.median(sc)), tunnel_rate=float(np.mean(sc >= 200)))
        tmp = out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)
        print(f"[EVAL{' ' + tag if tag else ''}] {run_dir.name} {res['tag']} env {env} upd {upd}: {mean:.2f} +/- {se:.2f} SE (HNS {hns:.1f}%) "
              f"scores={sc.tolist()} ({secs / 60:.1f} min)", flush=True)
        st = state.setdefault(run_dir.name, {})
        if "mlflow" not in st:
            st["mlflow"], st["run"] = mlflow_run(run_dir.name, a, proto, tag)
        if st["mlflow"]:
            try:
                extra = {"eval/median": res["median"], "eval/tunnel_rate": res["tunnel_rate"]} if tag else {}
                st["mlflow"].log_metrics({"eval/score": mean, "eval/se": se, "eval/hns": hns, **extra}, step=env)
            except Exception:
                pass
        if tag:  # another protocol: a summary of its own, model selection untouched
            summ_p = ckdir / f"eval_summary_{tag}.json"
            summ = json.loads(summ_p.read_text()) if summ_p.exists() else {"protocol": tag, "evals": []}
            summ["evals"] = sorted(summ["evals"] + [{k: res[k] for k in ("tag", "env_steps", "updates", "mean", "se",
                                                                         "median", "tunnel_rate")}],
                                   key=lambda r: r["updates"])
            tmp = summ_p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(summ, indent=1)); os.replace(tmp, summ_p)
            continue
        summ_p = ckdir / "eval_summary.json"
        summ = json.loads(summ_p.read_text()) if summ_p.exists() else {"best": None, "evals": []}
        summ["evals"] = sorted(summ["evals"] + [{k: res[k] for k in ("tag", "env_steps", "updates", "mean", "se")}],
                               key=lambda r: r["updates"])
        if summ["best"] is None or mean > summ["best"]["mean"]:
            summ["best"] = {k: res[k] for k in ("tag", "env_steps", "updates", "mean", "se", "checkpoint")}
            ck["eval_score"], ck["eval_scores"], ck["best"] = mean, sc.tolist(), mean
            ck.pop("eval_pending", None)
            tmpb = ckdir / "checkpoint_best.pt.tmp"; torch.save(ck, tmpb); os.replace(tmpb, ckdir / "checkpoint_best.pt")
        tmp = summ_p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(summ, indent=1)); os.replace(tmp, summ_p)
        final_done |= res["tag"] == "final"
        if final_done and st["mlflow"]:
            try:
                st["mlflow"].log_metric("best_eval", summ["best"]["mean"]); st["mlflow"].end_run()
            except Exception:
                pass
    if tag and state.get(run_dir.name, {}).get("mlflow"):  # one MLflow run per (run, protocol)
        try:
            state[run_dir.name]["mlflow"].end_run()
        except Exception:
            pass
        state.pop(run_dir.name)
    return final_done


def verify(ck_path: Path, device) -> int:
    ck, a, sc, secs = score(ck_path, device)
    ref = ck.get("eval_scores")
    print(f"re-scored {ck_path.name}: {sc.tolist()} ({secs / 60:.1f} min)")
    print(f"inline eval stored:        {ref}")
    if ref is None:
        print("no inline scores stored in this checkpoint; nothing to compare"); return 2
    same = np.allclose(np.asarray(ref, float), sc)
    print("MATCH" if same else f"DIFFERENT: mean {np.mean(ref):.2f} inline vs {sc.mean():.2f} deferred")
    return 0 if same else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--watch", action="store_true", help="poll for new checkpoints until each run's final one is scored")
    ap.add_argument("--poll", type=int, default=60, help="seconds between scans with --watch")
    ap.add_argument("--verify", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=None, help="protocol override: episodes (default: the run's)")
    ap.add_argument("--sims", type=int, default=None, help="protocol override: search simulations (default: the run's)")
    ap.add_argument("--sticky", type=float, default=0.0, help="protocol override: sticky-action probability")
    ap.add_argument("--flip-avg", action="store_true",
                    help="protocol override: average the search root with its mirror image (TODO B26 step 0)")
    ap.add_argument("--protocol", default=None, help="name for the override protocol's result files")
    ap.add_argument("--env-steps", type=lambda s: {int(x) for x in s.split(",")}, default=None,
                    help="only score the checkpoints at these env steps (comma list)")
    args = ap.parse_args(argv)
    proto = None
    if args.episodes or args.sims or args.sticky > 0 or args.flip_avg or args.protocol:
        proto = {"episodes": args.episodes, "sims": args.sims, "sticky": args.sticky, "flip": args.flip_avg,
                 "tag": args.protocol}
        args.watch = False  # protocol runs score saved checkpoints; there is no final checkpoint to wait for
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verify:
        return verify(args.verify, device)
    state, pending = {}, list(args.run_dirs)
    while pending:
        pending = [r for r in pending if not process_run(r, device, state, proto, args.env_steps)]
        if not args.watch:
            break
        if pending:
            time.sleep(args.poll)
    print("eval_ez_checkpoints: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
