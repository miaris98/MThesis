#!/usr/bin/env python3
"""Aggregate World on Rails training runs into one comparison table.

Reads each run directory's `wor_training_telemetry.csv` and `run_config.json` and
prints the numbers an ablation is actually decided on. Two things it does that reading
the CSVs by hand does not:

  - Groups runs by `run_label` and reports the spread across seeds. That spread is the
    noise floor: a component that moves the metric by less than it has not been shown
    to do anything, however good the mean looks.
  - Reports lateral and longitudinal error separately. The combined loss is
    `longitudinal + 3 x lateral`, and in the first CNN-vs-transformer comparison 95.5%
    of the gap was longitudinal while the metric that actually steers the car was
    nearly matched - a distinction the headline number hides.

Usage:
    python compare_wor_runs.py <run_dir> [<run_dir> ...]
    python compare_wor_runs.py --glob "experiments/checkpoints/wor_*"
    python compare_wor_runs.py --glob "runs/*" --baseline qwen30m_seed0 --csv out.csv
"""
import argparse
import csv
import glob as globlib
import json
import math
import os
import statistics
from typing import Dict, List, Optional

TELEMETRY_NAME = "wor_training_telemetry.csv"
CONFIG_NAME = "run_config.json"

#: Config keys worth showing as the "what differed" column, in display order.
DISTINGUISHING_KEYS = [
    "policy_arch", "backbone", "vision_grid", "pool_vision", "seed", "epochs",
    "batch_size", "lr_heads", "grad_clip", "warmup_frac", "decay_gates_and_norms",
    "route_points", "val_split", "lateral_loss_weight", "freeze_backbone",
]


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    """Float from a CSV cell, or None for blank/unparseable/non-finite."""
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def split_runs(rows: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """Splits a telemetry file into separate runs wherever the epoch counter resets.

    Runs append to the same CSV, so one file can hold several. The first shipped
    telemetry file held an 8-epoch run followed by a 50-epoch one, which is easy to
    average together by accident.
    """
    runs: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    for row in rows:
        try:
            epoch = int(row["epoch"])
        except (KeyError, ValueError):
            continue
        if current and epoch <= int(current[-1]["epoch"]):
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def summarize(run_dir: str) -> Optional[Dict]:
    """Reduces one run directory to a single row of comparable numbers."""
    csv_path = os.path.join(run_dir, TELEMETRY_NAME)
    if not os.path.isfile(csv_path):
        return None

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    runs = split_runs(rows)
    if not runs:
        return None
    epochs = runs[-1]  # the most recent run in the file

    config = {}
    config_path = os.path.join(run_dir, CONFIG_NAME)
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            config = {}

    # Select on held-out loss where it exists, else training loss - and say which,
    # because they are not comparable to each other.
    has_val = any(_f(r, "val_loss") is not None for r in epochs)
    key = "val_loss" if has_val else "total_loss"
    scored = [(r, _f(r, key)) for r in epochs]
    scored = [(r, v) for r, v in scored if v is not None]
    if not scored:
        return None
    best_row, best_value = min(scored, key=lambda pair: pair[1])

    prefix = "val_" if has_val else ""
    total_batches = sum(int(r.get("num_batches") or 0) for r in epochs)
    clipped = sum(int(float(r.get("clipped_grad_batches") or 0)) for r in epochs)
    nonfinite = sum(int(float(r.get("nonfinite_grad_batches") or 0)) for r in epochs)

    return {
        "run_dir": run_dir,
        "label": config.get("run_label") or os.path.basename(os.path.normpath(run_dir)),
        "config": config,
        "selected_on": key,
        "epochs": len(epochs),
        "best_epoch": int(best_row["epoch"]),
        "best_loss": best_value,
        "final_loss": scored[-1][1],
        "train_loss": _f(best_row, "total_loss"),
        "ade": _f(best_row, f"{prefix}wp_ade_m"),
        "fde": _f(best_row, f"{prefix}wp_fde_m"),
        "lateral": _f(best_row, f"{prefix}wp_lateral_error_m"),
        "longitudinal": _f(best_row, f"{prefix}wp_longitudinal_error_m"),
        "grad_norm": _f(epochs[-1], "grad_norm"),
        "clipped_frac": (clipped / total_batches) if total_batches else None,
        "nonfinite_batches": nonfinite,
        "samples_per_sec": _f(epochs[-1], "samples_per_sec"),
        "wall_time_s": _f(epochs[-1], "wall_time_s"),
        "seed": config.get("seed", _f(epochs[-1], "seed")),
    }


def _fmt(value, spec: str = ".4f", dash: str = "-") -> str:
    return dash if value is None else format(value, spec)


def print_runs_table(runs: List[Dict]) -> None:
    header = (f"{'run':<28}{'sel':>5}{'best':>9}{'ep':>4}{'ADE':>8}{'lat':>8}{'lon':>8}"
              f"{'|g|':>7}{'clip%':>7}{'nf':>4}{'s/s':>8}{'min':>7}")
    print(header)
    print("-" * len(header))
    for run in sorted(runs, key=lambda r: (r["label"], str(r["seed"]))):
        clip = run["clipped_frac"]
        wall = run["wall_time_s"]
        # Seed in the row name, or repeats of one configuration are indistinguishable.
        name = f"{run['label']}#{run['seed']}" if run["seed"] is not None else run["label"]
        print(f"{name[:27]:<28}"
              f"{('val' if run['selected_on'] == 'val_loss' else 'trn'):>5}"
              f"{_fmt(run['best_loss']):>9}"
              f"{run['best_epoch']:>4}"
              f"{_fmt(run['ade']):>8}"
              f"{_fmt(run['lateral']):>8}"
              f"{_fmt(run['longitudinal']):>8}"
              f"{_fmt(run['grad_norm'], '.2f'):>7}"
              f"{('-' if clip is None else format(100 * clip, '.0f')):>7}"
              f"{run['nonfinite_batches']:>4}"
              f"{_fmt(run['samples_per_sec'], '.0f'):>8}"
              f"{('-' if wall is None else format(wall / 60, '.0f')):>7}")


def print_groups_table(runs: List[Dict], baseline: Optional[str]) -> None:
    """One row per configuration, with the across-seed spread that makes it readable."""
    groups: Dict[str, List[Dict]] = {}
    for run in runs:
        groups.setdefault(run["label"], []).append(run)

    base_mean = None
    if baseline and baseline in groups:
        base_mean = statistics.mean(r["best_loss"] for r in groups[baseline])

    print(f"\n{'configuration':<28}{'n':>3}{'best mean':>11}{'spread':>9}"
          f"{'lat':>9}{'lon':>9}{'vs base':>10}")
    print("-" * 79)
    for label in sorted(groups):
        members = groups[label]
        losses = [r["best_loss"] for r in members]
        mean = statistics.mean(losses)
        spread = (max(losses) - min(losses)) if len(losses) > 1 else None
        lat = [r["lateral"] for r in members if r["lateral"] is not None]
        lon = [r["longitudinal"] for r in members if r["longitudinal"] is not None]
        delta = ""
        if base_mean is not None and label != baseline:
            pct = 100.0 * (mean - base_mean) / base_mean
            delta = f"{pct:+.1f}%"
        print(f"{label[:27]:<28}{len(members):>3}{mean:>11.4f}"
              f"{('-' if spread is None else format(spread, '.4f')):>9}"
              f"{(statistics.mean(lat) if lat else float('nan')):>9.4f}"
              f"{(statistics.mean(lon) if lon else float('nan')):>9.4f}"
              f"{delta:>10}")

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        worst = max((max(r["best_loss"] for r in v) - min(r["best_loss"] for r in v))
                    for v in multi.values())
        ref = statistics.mean(statistics.mean(r["best_loss"] for r in v) for v in multi.values())
        print(f"\nNoise floor: widest across-seed spread is {worst:.4f} "
              f"({100 * worst / ref:.1f}% of the mean).")
        print("Treat any difference smaller than this as unmeasured, not as an effect.")
    else:
        print("\n[Warning] Every configuration has a single seed, so no noise floor is "
              "available and no difference below is attributable. Run one config at "
              "2-3 seeds first.")


def print_config_diff(runs: List[Dict]) -> None:
    """Shows only the settings that actually differ between the runs being compared."""
    configs = [r["config"] for r in runs if r["config"]]
    if len(configs) < 2:
        return
    varying = [k for k in DISTINGUISHING_KEYS
               if len({json.dumps(c.get(k), default=str) for c in configs}) > 1]
    if not varying:
        print("\nAll compared runs share the same recorded configuration.")
        return
    print(f"\n{'run':<28}" + "".join(f"{k[:13]:>15}" for k in varying))
    print("-" * (28 + 15 * len(varying)))
    for run in sorted(runs, key=lambda r: (r["label"], str(r["seed"]))):
        cells = "".join(f"{str(run['config'].get(k, '-'))[:14]:>15}" for k in varying)
        name = f"{run['label']}#{run['seed']}" if run["seed"] is not None else run["label"]
        print(f"{name[:27]:<28}{cells}")


def write_csv(runs: List[Dict], path: str) -> None:
    fields = ["label", "seed", "selected_on", "best_loss", "final_loss", "best_epoch",
              "epochs", "ade", "fde", "lateral", "longitudinal", "grad_norm",
              "clipped_frac", "nonfinite_batches", "samples_per_sec", "wall_time_s",
              "run_dir"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            writer.writerow(run)
    print(f"\nWrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", help="Run directories to compare")
    parser.add_argument("--glob", default=None, help="Glob pattern selecting run directories")
    parser.add_argument("--baseline", default=None,
                        help="run_label to report other configurations relative to")
    parser.add_argument("--csv", default=None, help="Also write the summary to this CSV path")
    args = parser.parse_args()

    dirs = list(args.run_dirs)
    if args.glob:
        dirs.extend(sorted(p for p in globlib.glob(args.glob) if os.path.isdir(p)))
    # A directory may be passed directly or as its telemetry file.
    dirs = [os.path.dirname(d) if d.endswith(".csv") else d for d in dirs]
    dirs = list(dict.fromkeys(dirs))

    if not dirs:
        parser.error("no run directories given - pass paths or --glob")

    runs, skipped = [], []
    for run_dir in dirs:
        summary = summarize(run_dir)
        (runs if summary else skipped).append(summary or run_dir)

    if skipped:
        print(f"[Warning] Skipped {len(skipped)} directory/ies without usable "
              f"{TELEMETRY_NAME}: {', '.join(os.path.basename(s) for s in skipped[:5])}"
              f"{' ...' if len(skipped) > 5 else ''}\n")
    if not runs:
        print("No runs with usable telemetry were found.")
        return 1

    print_runs_table(runs)
    print_groups_table(runs, args.baseline)
    print_config_diff(runs)

    if any(r["selected_on"] == "total_loss" for r in runs):
        print("\n[Warning] Some runs are scored on TRAINING loss because they have no "
              "validation set. Those rows are not comparable to val-scored rows.")
    if args.csv:
        write_csv(runs, args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
