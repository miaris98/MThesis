"""Helper script to inspect Optuna study progress and print summary."""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna

def main():
    storage = "sqlite:///results/atari_qwen/optuna.db"
    try:
        studies = optuna.get_all_study_summaries(storage=storage)
    except Exception as e:
        print(f"Error accessing storage: {e}")
        return

    if not studies:
        print("No studies found in storage.")
        return

    for s in studies:
        print(f"=== Study: {s.study_name} ===")
        print(f"Total Trials: {s.n_trials}")
        if s.best_trial:
            print(f"Best Trial #{s.best_trial.number} with Return: {s.best_trial.value:.2f}")
            print(f"Best Params: {s.best_trial.params}")

        study = optuna.load_study(study_name=s.study_name, storage=storage)
        for t in study.trials:
            state = t.state.name
            val = f"{t.value:.2f}" if t.value is not None else "In Progress"
            step = max(t.intermediate_values.keys()) if t.intermediate_values else 0
            print(f"  Trial {t.number:2d}: {state:10s} | Eval Return: {val:>8s} | Step: {step:7d} | Preset: {t.params.get('model_preset', 'N/A')}")

if __name__ == "__main__":
    main()
