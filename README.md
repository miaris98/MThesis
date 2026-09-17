# MThesis

Qwen-style transformer policies for end-to-end autonomous driving in CARLA,
benchmarked against World on Rails.

## Layout

| Path | What lives here |
| ---- | --------------- |
| `src/` | The library: models, environments, rewards, training loops, metrics. Imported, not executed. |
| `scripts/` | Entry points. Every script is runnable *and* importable (see below). |
| `scripts/eval/` | Closed-loop evaluation, CARLA leaderboard / Bench2Drive runs, result aggregation. |
| `scripts/training/` | Training entry points and sweep launchers. |
| `scripts/analysis/` | Offline analysis of datasets, checkpoints and run telemetry. |
| `scripts/setup/` | Environment provisioning: CARLA, vast.ai, dataset downloads, server launch. |
| `scripts/sync/` | Experiment sync, checkpoint upload, MLflow dashboards and tunnels. |
| `scripts/carla/` | Standalone CARLA client utilities and connectivity checks. |
| `atari_qwen/` | Self-contained Atari PPO study for the same transformer architecture. |
| `tests/` | Unit tests. |
| `docs/` | `todo/` roadmaps, `guides/` setup and troubleshooting, `design/` design notes and the struggle log, `archive/` tombstones for files moved to external storage. |
| `data/` | Small curated inputs (route subsets, fixed splits). Not experiment output. |
| `patches/` | Byte-exact LF patches applied to vendored third-party checkouts. |

`Carla-utils/`, `checkpoints/`, `results/`, `weights/`, `mlruns/` and
`experiments/` are untracked working directories — see `.gitignore` and
[docs/guides/experiment_storage_guide.md](docs/guides/experiment_storage_guide.md).

## Running things

Everything is addressed from the repository root:

```bash
python scripts/training/train_wor.py --data_dir /workspace/dataset/wor_trajectories
python scripts/eval/eval_wor_closed_loop.py --checkpoint .../best_model.pth
bash   scripts/eval/run_leaderboard_official.sh <label> <arch> <ckpt> <routes.xml>
```

Each entry point puts the repository root on `sys.path` itself, so it works both
as `python scripts/<area>/<name>.py` and as `import scripts.<area>.<name>` — that
is what lets `scripts/eval/bench2drive_agent.py` reuse `scripts/eval/eval_wor.py`
while the CARLA leaderboard runs it with its own cwd from inside the Bench2Drive
checkout.

```bash
python -m pytest tests/ -q
```
