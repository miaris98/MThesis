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

### 3. Compute Utilization & Parallelism (GPU & CPU Co-Utilization)
Before triggering any training or evaluation run:
- **Mandatory GPU & CPU Co-Utilization**:
  - **GPU**: Fully saturate available GPU compute and VRAM on the target instance (e.g., RTX A4000 16GB, RTX 3090 24GB, 2x Tesla T4), sizing batch sizes and model contexts to maintain a safe ~10–15% VRAM headroom to prevent CUDA OOM crashes.
  - **CPU**: Never let system CPU cores sit idle during training or simulation. Fully leverage high-core CPUs (e.g., AMD EPYC 64-core on Vast.ai).
- **Default to Parallel Execution**:
  - **Vectorized Environments**: For RL / simulation workloads (Atari, CARLA), always vectorize environment stepping (`num_envs=16`, `32`, or `64` scaled to available CPU cores).
  - **Parallel Data Loaders**: In supervised or offline distillation (e.g., PDM-Lite / World-on-Rails), scale PyTorch DataLoader workers (`num_workers=4` to `8`+), enable `pin_memory=True`, and set `prefetch_factor=2` to ensure GPU compute is never starved by CPU data loading.
  - **Parallel Evaluations & Sweeps**: Run evaluation rollouts, seed replications, and hyperparameter trials in parallel across available ports/threads.
  - **No Sequential / Single-Threaded Execution**: Never default to single-threaded or sequential execution (`num_envs=1`, `num_workers=0`, or sequential seed loops) unless the user explicitly instructs to run sequentially or single-threaded for isolated debugging.

### 4. Progressive 4-Gate Scaling Protocol & Checkpoint Continuity
To prevent burning compute budget on flawed directions, silent algorithmic collapses (e.g. policy drifting to 96% NOOP, reward plateaus, trajectory divergence), or AI agent misdirections:
- **Gate 1: Smoke & Mechanics POC**: 3k–5k steps / 1 epoch (<$0.03). Sanity check syntax, gradient flow, action entropy > 1.0. Saves `ckpt_gate1.pth`.
- **Gate 2: Early Dynamics & Signal Validation**: Steps 5k–20k / Epochs 1–5 (<$0.15–$0.40). Validates score > 0, value loss decrease, CARLA ADE < 0.45m / Lat Err < 8cm. Saves `ckpt_gate2.pth`.
- **Gate 3: Mid-Scale Stability & Robustness**: Steps 20k–50k / Epochs 5–20 (<$0.40–$1.50). Checks 4-way balance, no late-stage collapse, CARLA Lat Err < 4cm. Saves `ckpt_gate3.pth`.
- **Gate 4: Full Benchmark Production & Two-Stage Archive**: Steps 50k–100k / Epochs 20–50. Full literature comparison and final sync.
- **Mandatory Checkpoint Continuity**: Every gate MUST save a complete resumable state (`model`, `optimizer`, `lr_scheduler`, `scaler`, step/epoch). When scaling from Gate $N \to N+1$, **always resume using `--resume_from`**; never retrain from scratch or discard prior compute. Never recommend full Gate 4 runs without Gate 1–2 empirical validation.
