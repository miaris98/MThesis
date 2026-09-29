# scripts/

Entry points and operations tooling. Library code lives in `src/` and `atari_qwen/`. Before adding a new
`check_*.py` or `pull_*.py` for a specific run, use a generic tool below: most of the one-offs removed on
2026-09-28 were copies of these with a host or run name hard-coded.

## Portability: sizes and paths come from the box, not the script

| Helper | What it gives | Used by |
|---|---|---|
| `lib/machine.py` | The container's real CPU quota (cgroup v1/v2), `pids.max`, memory limit, GPUs (PCI order, VRAM, bf16), and job sizes derived from them: DataLoader workers, EZ-V2 Ray CPUs, CARLA lanes (5 cores + 7.5 GB VRAM each, 15 % VRAM headroom), eval OMP threads. `--export` prints shell exports. | `train_wor.py`, `build_feature_cache.py`, `ezv2_10k_seeds.sh`, `box.py` |
| `lib/common.sh` | Overridable paths (`WORKSPACE`, `MTHESIS_ROOT`, `CARLA_DIR`, `GARAGE`, `VENV_CARLA`, `B2D_OUT`, `GUARDIAN_LOGS`, `CHECKPOINTS`, `PYTHON3`), `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and one tested copy of `desc` / `kill_tree` / `carla_pids_on_port` / `clear_carla_ports` / `json_ok` / `b2d_progress` / `b2d_records`. | all Bench2Drive eval scripts, `start_carla.sh` |
| `setup/remote_config.py` | SSH host/port/user/venvs from the git-ignored `.remote` file (env overrides `REMOTE_*`). | `box.py` and every script that SSHes to a box |

Rules these encode, learnt the hard way:
- Never size from `os.cpu_count()` or `multiprocessing.cpu_count()`: that is every host CPU, not the quota (S-086).
- Never kill by a bare name pattern (`pkill -f CarlaUE4`, `pkill -f <run>`). Kill a pid's tree (`kill_tree`) or a
  port's servers (`clear_carla_ports`) (S-013, S-091, S-099).
- Write checkpoints atomically (tmp + `os.replace`), and never sync or push `*.tmp` files.

A box with a different layout needs environment variables only, e.g.
`WORKSPACE=/data GARAGE=/opt/carla_garage bash scripts/eval/launch_b2d20.sh run`.

## Directory guide

**`eval/`: closed-loop evaluation**
- Current Bench2Drive pipeline:
  - `launch_b2d20.sh` (setup / preflight / run; setup applies `setup/patch_bench2drive.py`);
  - `queue_A9c.sh` (one lane, a job list);
  - `b2d_guardian.sh` (relaunch until done, capped attempts);
  - `eval_watchdog.sh` (kills stuck attempts);
  - `run_bench2drive.sh` (one arm);
  - `bench2drive_agent.py` / `wor_official_b2d_agent.py` (agents).
- Results: `check_leaderboard_results.py` (refuses crashed/unrecognised statuses), `merge_leaderboard_results.py`
  (retry passes), `compare_closed_loop.py`, `scenario_breakdown.py`.
- Older tiers (eval_tiers_design.md): `run_fast_eval.sh`, `run_leaderboard_official.sh`,
  `run_leaderboard_resilient.sh`, `run_closed_loop_*.sh`, `eval_wor_closed_loop.py`, `eval_wor.py`,
  `record_eval_video.py`, `run_tcp.sh`, `run_town_probe.sh`.
- Atari: `eval_ezv2_checkpoint.py`.

**`analysis/`: offline analysis of results and data**
- `b2d_run_noise.py`: run-to-run noise, fixed-set vs route-population CIs, DS decomposition, collision lottery
  (S-100, S-101). `--preset b2d20_2026-09-28` reproduces those numbers.
- `route_overlay_slope_check.py`: flat-ground overlay error on hills (S-100).
- Dataset and target audits: `analyze_wor_targets.py`, `audit_route_leakage.py`, `validate_nav_signal.py`,
  `inspect_pdm_measurements.py`, `check_val_noise.py`, `score_checkpoints_on_val.py`, `compare_wor_runs.py`.

**`setup/`: provisioning**
- `box.py`: `probe` / `status` / `env` on the `.remote` box, including its real limits.
- `setup_vastai.sh`, `setup_carla_eval.sh`, `setup_tfpp_env.sh`, `download_pdm_lite.py`, `apply_easycarla_patch.sh`.
- `start_carla.sh` (per-port; `KILL_ALL_CARLA=1` for a full reset) and `start_carla_multi.sh` (per-port log name).
- `patch_bench2drive.py`: Bench2Drive's evaluator killed every CARLA on its GPU after one lane crashed. The patch
  makes it kill only its own server (idempotent).

**`sync/`: two-stage sync** (external disk first, Hugging Face second)
- `sync_experiments.py` (pull + MLflow merge) and `pull_loop.py` (periodic pull for a live run).
- `live_sync_loop.sh`: periodic tar-over-ssh backup into a staging directory, merged only when the stream arrived
  complete.
- `hf_push_checkpoints.py`: skips in-progress `*.tmp` files.
- `build_external_graph.py`, `download_from_vastai.{py,ps1}`, MLflow and dashboard helpers.
- `guardian.sh` delegates to `eval/b2d_guardian.sh`.

**`training/`**
- `train_wor.py` (CARLA distillation; worker count from the CPU quota) and `build_feature_cache.py`.
- `run_multi_carla_training.sh`, `run_wor_sweep.sh`, `run_seed_replication_arms.sh`, `launch_retrain_v2.sh`.
- `ezv2_setup_blackwell.sh` + `ezv2_10k_seeds.sh` (official EZ-V2 seeds; Ray CPUs from the quota and pids.max).
- `train_rl_agent.py` (EasyCARLA RL).

## Removed on 2026-09-28 (pending deletion in the working tree)

60 one-off scripts (per-run status checks, launchers for the retired `train_mcts_offpolicy` trainer, pulls with
hard-coded hosts, the phase-4 eval family, `scratch/`, root-level BC checks) are archived with SHA-256 and reasons
in `E:\MThesis_EXP\local_repo_logs_20260928\scripts_cleanup_20260928\MANIFEST.md`. Their replacements are the
generic tools above.
