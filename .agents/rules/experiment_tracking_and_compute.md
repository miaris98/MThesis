---
trigger: always_on
description: Rules for MLflow tracking, two-stage syncing (External drive first, then HuggingFace), and maximizing machine compute utilization with safe OOM headroom.
---

## Experiment Tracking, Syncing & Machine Utilization Rules

### 1. Mandatory MLflow Experiment Tracking
Whenever creating or updating scripts for training, testing, or evaluating:
- **Always integrate MLflow experiment tracking**.
- Log all hyperparameters/config, step/epoch metrics (losses, ADE, lateral error, rewards, episode return, SPS), and final artifacts/checkpoints.
- Preserve local and external tracking store compatibility (`paths.mlruns_dir()` with `MLFLOW_ALLOW_FILE_STORE=true`).

### 2. Two-Stage Syncing Pipeline (External First -> Hugging Face Second)
All artifacts, checkpoints, and evaluation results must follow a strict two-stage sync pipeline:
1. **Stage 1 — External Archive First (`E:\MThesis_EXP`)**:
   - Sync all logs, telemetry CSVs, model weights, and MLflow runs to `E:\MThesis_EXP` first using `scripts/sync/sync_experiments.py` or direct export.
   - Update the external knowledge graph and catalog via `python scripts/sync/build_external_graph.py`.
2. **Stage 2 — Hugging Face Second**:
   - Push verified champion checkpoints, evaluation summaries, and gameplay/rollout videos to Hugging Face using the credentials in the local `.env` file (`HF_TOKEN`) via existing sync scripts (e.g., `scripts/sync/hf_push_checkpoints.py` or `atari_qwen/training/hf_sync.py`).

### 3. Compute Utilization & Parallelism
Before triggering any training or evaluation run:
- **Maximize Hardware Utilization**: Fully leverage available GPU VRAM, CPU cores, and system memory on the target machine (e.g., RTX A4000 16GB, EPYC 64-core).
- **Safe OOM Headroom**: Size batch sizes, rollout steps, and replay buffer capacities to leave a safe ~10–15% VRAM headroom to prevent CUDA Out-Of-Memory (OOM) crashes.
- **Default to Parallel Execution**: Run workloads in parallel by default (e.g., vectorized simulation environments `num_envs=16` or `32`, multi-worker data loaders `num_workers=4` to `8`, or parallel evaluation trials) unless explicitly instructed by the user to run sequentially or single-threaded.
