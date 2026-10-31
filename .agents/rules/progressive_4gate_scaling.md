---
trigger: always_on
description: Progressive 4-Gate scaling protocol for cost protection and mandatory checkpoint continuity across Atari and CARLA experiments.
---

# Progressive 4-Gate Scaling Protocol (Cost Protection & Checkpoint Continuity)

To prevent burning compute budget on flawed directions, silent algorithmic collapses (e.g. policy drifting to 96% NOOP, attention one-hot collapse, reward plateaus, trajectory divergence), or AI agent misdirections, all experiments must strictly advance through **4 Progressive Iterative Gates**.

### Principle: Checkpoint Continuity
- **Never restart from scratch** when progressing to higher gates.
- Every gate must serialize complete resumable state (`model`, `optimizer`, `lr_scheduler`, `scaler`, `step`/`epoch` index, and for Atari the matched `checkpoint_latest.pt` + `replay_latest.npz` pair).
- Progressing from Gate `N -> N+1` must resume directly from Gate `N`'s checkpoint using `--resume_from <path>`. (For Atari 30k screens, resume to 100k specifically from `checkpoint_env30000_upd28004.pt` + `replay_latest.npz`).

---

### Gate 1: Smoke & Mechanics POC (Sanity Check)
- **Scale**:
  - Atari: 3,000 – 5,000 environment steps (~5–8 min, < $0.03).
  - CARLA: 1 epoch or 500 mini-batches (~8–10 min, < $0.04).
- **Goal**: Confirm syntax, shape compatibility, gradient flow, loss finiteness, and absence of immediate single-action or attention collapse.
- **Pass Criteria**:
  - Loss is finite and trending down.
  - Action distribution entropy > 1.0 (no single action > 85%).
  - Zero CUDA OOM errors, VRAM headroom >= 15%.
- **Action on Failure**: Abort immediately. Zero compute budget burned.
- **Artifact**: Save `ckpt_gate1.pth`.

---

### Gate 2: Early Dynamics & Signal Validation (Convergence Check)
- **Scale**: Resumes from `ckpt_gate1.pth`.
  - Atari: Step 5,000 -> 10,000–20,000 steps (~30 min, < $0.15; 10k-step screens in spare headroom judged against human score 30.5 on Breakout and EZ-V2 10k 9.8 / 20k 37.2; check `mixer_diag.py` attention entropy).
  - CARLA: Epoch 1 -> 5 epochs (~1.5 hours, < $0.40).
- **Goal**: Validate that the agent is learning non-trivial signal and interacting purposefully with the environment.
- **Pass Criteria**:
  - Atari: Evaluation score > 0 (or distinct rally dynamic), value prediction error dropping, reward prediction correlation positive, attention entropy healthy.
  - CARLA: Held-out validation ADE < 0.45m, Lateral Error < 0.08m (8 cm), Waypoint Loss < 0.60.
- **Action on Failure**: Stop. Inspect value/policy balance, mixer weight norms, or latent dynamics. Do not scale until the failure mode is understood.
- **Artifact**: Save `ckpt_gate2.pth`.

---

### Gate 3: Mid-Scale Stability & Robustness Check
- **Scale**: Resumes from `ckpt_gate2.pth`.
  - Atari: Step 20,000 -> 30,000–50,000 steps (~1.5–3.5 hours, < $0.40–$0.90). Note: at 30k, stop only on gross failure (>=3 seeds below 50% of baseline), never on narrow margin stops.
  - CARLA: Epoch 5 -> 20 epochs (`--epochs 50 --stop_epoch 20`, screened on the 80-route stratified proxy).
- **Goal**: Check for late-stage stability, absence of policy drift or catastrophic forgetting, and multi-seed consistency.
- **Pass Criteria**:
  - Atari: Score sustained or advancing, 4-way action balance maintained, search depth calibrated.
  - CARLA: Held-out validation ADE < 0.35m, Lateral Error < 0.04m (4 cm), competitive DS on the 80-route proxy (within 7 DS of reference).
- **Action on Failure**: If drift occurs, rollback to `ckpt_gate2.pth` with targeted regularization adjustments rather than restarting from step 0.
- **Artifact**: Save `ckpt_gate3.pth` (`model_epoch_020.pth` / `checkpoint_env30000_upd28004.pt`).

---

### Gate 4: Full Benchmark Production & Two-Stage Archive
- **Scale**: Resumes from `ckpt_gate3.pth`.
  - Atari: Step 30,000/50,000 -> 100,000 env steps (`--updates 120000` final, >=5 seeds for headline claims).
  - CARLA: Epoch 20 -> 50 epochs, followed by full 220-route Bench2Drive closed-loop evaluation (`Low` default + `Epic` column for Low-crashing routes, >=3 runs for claims).
- **Goal**: Final research champions and publication-grade metrics.
- **Mandatory Finalization Pipeline**:
  1. **Stage 1 (External Archive First)**: Sync all checkpoints, resumable pairs, telemetry CSVs, logs, and MLflow runs to `E:\MThesis_EXP` with verified `sha256`.
  2. **Stage 2 (Google Drive / HF Second)**: Push champion weights, resumable pairs, and rollout videos to Google Drive (`MThesis_backups/` via `scripts/sync/gdrive_push_dir.sh`, verifying `PUSH_VERIFY ... bad 0 errors 0`).
