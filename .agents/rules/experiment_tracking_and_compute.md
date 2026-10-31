---
trigger: always_on
description: Rules for MLflow tracking, two-stage syncing (E:\MThesis_EXP first, Google Drive / HF second), backup verification, Atari resumable pairs, and GPU/CPU compute utilization.
---

# Experiment Tracking, Two-Stage Syncing, Backup Verification & Compute Utilization

## 1. Mandatory MLflow Experiment Tracking & Checkpoint Cadence
Whenever creating or updating scripts for training, testing, or evaluating:
- **Always integrate MLflow experiment tracking** (`--use_mlflow 1`). `run_config.json` must never have `"use_mlflow": false` for any real run.
- Log all hyperparameters/config, step/epoch metrics (losses, ADE, lateral error, rewards, episode return, SPS), and final artifacts/checkpoints.
- Preserve local and external tracking store compatibility (`paths.mlruns_dir()` with `MLFLOW_ALLOW_FILE_STORE=true`).
- **Bounded Checkpoint Cadence**:
  - Save a resumable checkpoint at least every 3 epochs (`train_wor.py --save_freq 3`) or every 2,500 updates for step-based Atari trainers.
  - Always maintain both `latest_*` (every save) and `best_*` (overwritten only when the tracked metric improves via `is_best`).
  - For frozen-backbone CARLA runs, always copy `frozen_backbone.pth` alongside `model_epoch_*.pth` and `run_config.json`.

---

## 2. Two-Stage Syncing Pipeline (`E:\MThesis_EXP` First -> Google Drive Second)
All artifacts, checkpoints, and evaluation results must follow a strict two-stage sync pipeline:
1. **Stage 1 — External Archive First (`E:\MThesis_EXP`)**:
   - Sync all logs, telemetry CSVs, model weights, and MLflow runs to `E:\MThesis_EXP` first using `scripts/sync/sync_experiments.py` or chunked verified pulls.
   - Update the external knowledge graph and catalog via `python scripts/sync/build_external_graph.py`.
   - **Never treat `C:` staging as a backup**: `C:` has limited free space and is only a temporary transit area. A box is only backed up when `E:\MThesis_EXP` and Stage 2 hold verified copies. If `E:` drops ("semaphore timeout" / "Permission denied"), tell the user immediately, keep retrying, and clean `C:` staging after `E:` is verified.
2. **Stage 2 — Google Drive Second (replaces Hugging Face as primary Stage 2, 2026-10-19)**:
   - Push verified champion checkpoints, evaluation summaries, resumable pairs, and gameplay/rollout videos to Google Drive (`MThesis_backups/`) with `rclone`, from the boxes, via `scripts/sync/gdrive_push_dir.sh DIR SUB` (checksum-verified, prints `PUSH_VERIFY`). Full guide: `docs/ops/google_drive_sync.md`.
   - Credential: `.gdrive_rclone.conf` (repo root, gitignored; lands in `/dev/shm/gdrive_rclone.conf` on boxes, RAM only).
   - `fleet_monitor/sync_boxes.sh` and `finisher.sh` push to Drive by default (`STAGE2=hf` / `FIN_TARGET=hf` for the legacy HF relay); `live_sync_loop.sh` is Stage 1 only.
   - **Never claim a backup is on Drive until a box-side `PUSH_VERIFY ... bad 0 errors 0` line has been seen.**
   - Hugging Face scripts stay for restoring old runs or when explicitly used; note that the HF private relay (`Miaris/mthesis-relay`) hit its ~41.5 GB storage cap on 2026-10-27, so always check free space before any HF upload and batch HF API tree checks with a single recursive `list_repo_tree` (limit: 1,000 req / 5 min).

---

## 3. Backup Integrity, Resumable Pairs & Pre-Destroy Verification
- **Immediate Backup on Completion**: The moment a training run completes, immediately pull `best_model.pth`, `latest_model.pth`, `run_config.json`, and telemetry CSVs to `E:\MThesis_EXP` and Stage 2 before starting evaluation or any next step.
- **Atari Resumable Runs Backed Up as a Pair (S-140)**:
  - An Atari seed resumes **only** from the matched pair `checkpoint_latest.pt` (~117 MB) + `replay_latest.npz` (1.2–3.0 GB), saved together every 2,500 updates (~25 min). Never pass `--start-step` with weights-only.
  - Back up both to `E:` and Drive/HF on a wall-clock schedule (every 30 min) with per-box timeouts. Treat any repeated `"WRITE FAILED"` or `"MISMATCH"` log line as an active failure to fix immediately.
  - In short-notice saves before a destroy deadline, `gzip -1` uncompressed uint8 replay buffers on the box (~28x shrinkage) and start the final save with **at least 30 minutes of verified throughput margin** (begin final save prep >=90 min before deadline).
- **Measured Box Transfer Rates (B69, 2026-10-27)**:
  - Box -> `E:` over SSH: 0.2–0.5 MB/s per single stream (~0.9–1.0 MB/s across 4 streams).
  - Box -> Google Drive (`rclone`): ~0.4 MB/s end-to-end with MD5 check.
  - At 0.3–0.5 MB/s, a raw 3 GB replay pair takes 2–3 hours, so always compare measured `box_provision_probe.sh` speeds (>=2 MB/s required) against pair size and gzip replays when needed.
- **Large Copies & Pre-Destroy Checks**:
  - Single large `scp` or `ssh cat` transfers can truncate silently. Use snapshot + `split -b 100M` chunked pulls with per-chunk and whole-file `sha256` verification (`scratchpad/pull_final.sh` pattern), and read the explicit `VERIFIED` / `MISMATCH` verdict line (not just `tail`).
  - Before declaring a box safe to destroy: (1) verify the remote file listing is non-empty and print file counts and total bytes on both sides; (2) check that the newest checkpoint has its full expected byte size (not a 512-byte tar multiple truncation); (3) verify `sha256` hashes on the box *after* copying; (4) check `.sync_status.log` timestamps and restart sync loops before the 2-hour background task limit (~1h 45m).

---

## 4. Compute Utilization & Parallelism (GPU & CPU Co-Utilization)
Before triggering any training or evaluation run:
- **Mandatory GPU & CPU Co-Utilization**:
  - **GPU**: Fully saturate available GPU compute and VRAM on the target instance, sizing batch sizes (`--auto_batch_size 1 --vram_headroom_mb 2000`) and contexts to maintain a safe ~10–15% VRAM headroom (for box-side tests alongside two running jobs on a 5090, cap test VRAM at 12%).
  - **CPU**: Size jobs from `scripts/lib/machine.py` (cgroup quota and `pids.max`), never `os.cpu_count()`.
- **Default to Parallel Execution**:
  - **Vectorized Environments**: Always vectorize RL/simulation stepping (`num_envs=16`, `32`, or `64`).
  - **Parallel Data Loaders**: Scale PyTorch DataLoader workers (`num_workers=4` to `8`+), `pin_memory=True`, `prefetch_factor=2`.
  - **No Sequential Execution**: Never default to `num_envs=1`, `num_workers=0`, or sequential seed loops unless explicitly requested for isolated debugging.
  - **No Idle Paid Boxes**: A paid box with no running job is idle — flag it in the 10-minute check and immediately start the next prepared job or eval-only filler queue.
