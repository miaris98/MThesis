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
| `lib/pc.sh` | PC-side (Git Bash) defaults: `MTHESIS_EXP_ROOT` (the archive, default `E:/MThesis_EXP`; the same variable `src/config/paths.py` reads) and `EXP` (its Git Bash form for scp), `GDRIVE_ROOT`, `vast_cli`, `rclone_cli`. | `box_deploy.sh`, `box_provision_probe.sh`, `fleet_monitor/sync_boxes.sh`, the token scripts |
| `lib/vast_cli.py` | The `vastai` CLI: `$VAST_CLI`, else PATH, else a conda env under the home folder (no per-user path in code). | `fleet_monitor.py`, `run_status_report.py`, `vast_fleet.py`, `vast_search_pytorch.py` |
| `lib/chain.sh` | The box chains' status file, `await` / `step` / `wait_files` / `box_tests` / `render_test` / `lanes_done` / `trained_list` (one copy; was pasted into every chain). | `ops/*_chain.sh` |
| `lib/ez_train.sh` | The Atari trainer environment, NVIDIA MPS start and the resumable 5-try loop (`checkpoint_latest.pt` + replay pair). | `atari/run_b66_*.sh`, `run_b67_qkn.sh`, `run_b69.sh`, `run_b100k_qnowd.sh` |
| `src/config/carla_client.py` | Puts CARLA's client egg (and `agents.*`) on `sys.path` from `CARLA_ROOT`, else `$WORKSPACE/carla`. | `src/envs/base_env.py`, `eval/eval_wor.py`, `eval/record_eval_video.py`, `carla/*.py` |

Rules these encode, learnt the hard way:
- Never size from `os.cpu_count()` or `multiprocessing.cpu_count()`: that is every host CPU, not the quota (S-086).
- Never kill by a bare name pattern (`pkill -f CarlaUE4`, `pkill -f <run>`). Kill a pid's tree (`kill_tree`) or a
  port's servers (`clear_carla_ports`) (S-013, S-091, S-099).
- Write checkpoints atomically (tmp + `os.replace`), and never sync or push `*.tmp` files.

A box with a different layout needs environment variables only, e.g.
`WORKSPACE=/data GARAGE=/opt/carla_garage bash scripts/eval/launch_b2d20.sh run`.

No script spells out a machine-specific path (2026-11-22): box paths are `$WORKSPACE/...` in shell (`: "${WORKSPACE:=/workspace}"`) and
`src.config.paths.WORKSPACE` in Python; PC-side scripts that send commands to a box use `$RWS` (`REMOTE_WORKSPACE`); the archive is
`MTHESIS_EXP_ROOT` (`exp_root()` in Python, `$EXP` in Git Bash, which is also how usage examples write it); hosts and ports come from
`.remote`, the command line or the git-ignored `ops/fleet_monitor/boxes.txt`. `tests/test_scripts_*.py` run the shell helpers.

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
  `run_leaderboard_resilient.sh`, `run_closed_loop_arms.sh` (`ROUTES=random|official`, training seed as an argument), `eval_wor_closed_loop.py`, `eval_wor.py`,
  `record_eval_video.py`, `run_tcp.sh`, `run_town_probe.sh`.
- Atari: `eval_ezv2_checkpoint.py`.

**`analysis/`: offline analysis of results and data**
- `b2d_run_noise.py`: run-to-run noise, fixed-set vs route-population CIs, DS decomposition, collision lottery
  (S-100, S-101). `--preset b2d20_2026-09-28` reproduces those numbers.
- `route_overlay_slope_check.py`: flat-ground overlay error on hills (S-100).
- Dataset and target audits: `analyze_wor_targets.py`, `audit_route_leakage.py`, `validate_nav_signal.py`,
  `inspect_pdm_measurements.py`, `check_val_noise.py`, `score_checkpoints_on_val.py`, `compare_wor_runs.py`.

**`setup/`: provisioning**
- `box.py`: `probe` / `status` / `env` on the `.remote` box, including its real limits (it replaced the per-run `check_*.py`).
- `vast_fleet.py` (list / select into `.remote`), `vast_search_pytorch.py` (offers).
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

## Removed on 2026-11-22

65 scripts: the 17 one-offs the 2026-10-11 audit had queued, the per-run `setup/check_*.py` family (replaced by `box.py`), September
pulls and uploads with hard-coded hosts and paths, the phase-4 eval family, the `live_2026-10-10/` snapshot, and copies merged into
one script (`run_closed_loop_*`, `launch_lanes_nokill.sh`, `a86_chain.sh`, the `*_b100k_*` pair scripts, the four `*_deploy.sh` now
manifests for `ops/box_deploy.sh`). Verbatim copies with sha256 in `E:\MThesis_EXP\local_repo_logs_20261122etired_scripts\MANIFEST.md`;
`git checkout <commit> -- <path>` restores any of them.
