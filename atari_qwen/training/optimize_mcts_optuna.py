"""Optuna search over the off-policy MCTS trainer's first-20k-step convergence (S-055).

Goal: match or beat EfficientZero's Breakout score at 20k env steps. Each trial runs
train_mcts_offpolicy.py as a subprocess (clean GPU memory per trial, a crash fails one trial
not the study), streams its log, reports every "[EVALUATION] Step N | MCTS Score: X" to Optuna,
and is killed early when the MedianPruner says it is behind the other trials at the same step.

Search space = the improvements from the EfficientZero gap analysis (reanalyze, prioritized
replay, update budget) plus the optimizer/loss knobs they interact with. All trials use one seed
so trials differ only by config; the top configs must be re-run on other seeds before any claim.

usage (on the box):
  python atari_qwen/training/optimize_mcts_optuna.py --n-trials 50 --n-jobs 2 \
      --storage sqlite:////workspace/optuna_mcts.db --target-score <EZ 20k score>
"""
import argparse
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER = REPO_ROOT / "atari_qwen" / "training" / "train_mcts_offpolicy.py"
EVAL_RE = re.compile(r"\[EVALUATION\] Step\s+([\d,]+) \| MCTS Score: ([-\d.]+)")

# Current Gate-2 recipe (32 sims: 11.00 / 10.70 at 20k on seeds 1 / 2), enqueued as trial 0 so
# every other trial is judged against it inside the same study.
BASELINE = dict(num_simulations=32, lr=2.5e-4, replay_ratio=0.5, batch_size=128,
                reanalyze_ratio=0.0, priority_alpha=0.0, target_tau=0.005, ent_coef=0.01,
                value_loss_weight=0.25, consistency_loss_weight=0.5)

_slot_lock = threading.Lock()
_free_slots: list = []


def sample(trial: optuna.Trial) -> dict:
    return dict(
        num_simulations=trial.suggest_categorical("num_simulations", [16, 32]),
        lr=trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        replay_ratio=trial.suggest_categorical("replay_ratio", [0.5, 1.0, 2.0]),
        batch_size=trial.suggest_categorical("batch_size", [128, 256]),
        reanalyze_ratio=trial.suggest_categorical("reanalyze_ratio", [0.0, 0.1, 0.25]),
        priority_alpha=trial.suggest_categorical("priority_alpha", [0.0, 0.6, 1.0]),
        target_tau=trial.suggest_float("target_tau", 0.002, 0.02, log=True),
        ent_coef=trial.suggest_float("ent_coef", 1e-3, 2e-2, log=True),
        value_loss_weight=trial.suggest_categorical("value_loss_weight", [0.25, 0.5, 1.0]),
        consistency_loss_weight=trial.suggest_categorical("consistency_loss_weight", [0.5, 1.0, 2.0]),
    )


def make_objective(args):
    def objective(trial: optuna.Trial) -> float:
        p = sample(trial)
        with _slot_lock:
            cores = _free_slots.pop()
        name = f"trial_{trial.number:03d}"
        log_path = Path(args.out_dir) / f"{name}.log"
        cmd = [
            "taskset", "-c", cores, sys.executable, "-u", str(TRAINER),
            "--env-id", "BreakoutNoFrameskip-v4", "--total-steps", str(args.total_steps),
            "--num-envs", "8", "--seed", str(args.seed),
            "--eval-interval", str(args.eval_interval), "--eval-episodes", str(args.eval_episodes),
            "--ckpt-interval", "5000", "--min-replay-size", "1000",
            "--num-simulations", str(p["num_simulations"]), "--eval-simulations", str(p["num_simulations"]),
            "--lr", str(p["lr"]), "--replay-ratio", str(p["replay_ratio"]),
            "--batch-size", str(p["batch_size"]), "--reanalyze-ratio", str(p["reanalyze_ratio"]),
            "--priority-alpha", str(p["priority_alpha"]), "--target-tau", str(p["target_tau"]),
            "--ent-coef", str(p["ent_coef"]), "--value-loss-weight", str(p["value_loss_weight"]),
            "--consistency-loss-weight", str(p["consistency_loss_weight"]),
            "--log-dir", str(Path(args.out_dir) / name), "--run-label", f"optuna_{name}",
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, MLFLOW_ALLOW_FILE_STORE="true",
                   PYTHONPATH=str(REPO_ROOT))
        started = time.time()
        last = None
        try:
            with open(log_path, "w") as logf:
                logf.write(f"# params {p}\n# cmd {' '.join(cmd)}\n")
                logf.flush()
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, cwd=str(REPO_ROOT), env=env)
                for line in proc.stdout:
                    logf.write(line)
                    logf.flush()
                    m = EVAL_RE.search(line)
                    if m:
                        step, score = int(m.group(1).replace(",", "")), float(m.group(2))
                        last = score
                        trial.report(score, step)
                        trial.set_user_attr(f"score_{step}", score)
                        if step < args.total_steps and trial.should_prune():
                            proc.kill()
                            proc.wait()
                            raise optuna.TrialPruned(f"pruned at step {step} with score {score}")
                    if time.time() - started > args.trial_timeout_h * 3600:
                        proc.kill()
                        proc.wait()
                        raise optuna.TrialPruned(f"timeout after {args.trial_timeout_h} h")
                rc = proc.wait()
            if rc != 0 or last is None:
                raise RuntimeError(f"trainer exited rc={rc} without a final evaluation (see {log_path})")
            trial.set_user_attr("wall_clock_h", (time.time() - started) / 3600)
            return last
        finally:
            with _slot_lock:
                _free_slots.append(cores)
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=2, help="Concurrent trials (each ~4 GB VRAM, 8 env workers)")
    ap.add_argument("--cores-per-job", type=int, default=10)
    ap.add_argument("--first-core", type=int, default=0)
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--total-steps", type=int, default=20_000)
    ap.add_argument("--eval-interval", type=int, default=5_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trial-timeout-h", type=float, default=3.0)
    ap.add_argument("--target-score", type=float, default=None,
                    help="Stop the study once a completed trial reaches this 20k-step score (EfficientZero's)")
    ap.add_argument("--storage", type=str, default="sqlite:////workspace/optuna_mcts.db")
    ap.add_argument("--study-name", type=str, default="S055_mcts_first20k")
    ap.add_argument("--out-dir", type=str, default="/workspace/optuna_mcts")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for j in range(args.n_jobs):
        a = args.first_core + j * args.cores_per_job
        _free_slots.append(f"{a}-{a + args.cores_per_job - 1}")

    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage, direction="maximize", load_if_exists=True,
        sampler=TPESampler(seed=args.seed, n_startup_trials=8),
        # No pruning until 6 trials have finished, and never before the 10k evaluation: at 5k
        # nearly every trial is near random, so pruning there would be noise.
        pruner=MedianPruner(n_startup_trials=6, n_warmup_steps=args.eval_interval * 2 - 1),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE)

    callbacks = []
    if args.target_score is not None:
        def stop_at_target(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value >= args.target_score:
                print(f"*** trial {trial.number} reached {trial.value} >= target {args.target_score}; stopping")
                study.stop()
        callbacks.append(stop_at_target)

    remaining = max(0, args.n_trials - len([t for t in study.trials if t.state.is_finished()]))
    study.optimize(make_objective(args), n_trials=remaining, n_jobs=args.n_jobs,
                   callbacks=callbacks, catch=(RuntimeError,))

    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"finished: {len(done)} complete, "
          f"{len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])} pruned")
    for t in sorted(done, key=lambda t: t.value, reverse=True)[:5]:
        print(f"trial {t.number}: {t.value:.2f}  {t.params}")


if __name__ == "__main__":
    main()
