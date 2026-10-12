## Progressive 4-Gate Scaling Protocol (Cost Protection & Checkpoint Continuity)

To prevent burning compute budget on flawed directions, silent algorithmic collapses (e.g. policy drifting to 96% NOOP, reward plateaus, trajectory divergence), or AI agent misdirections, all experiments must strictly advance through **4 Progressive Iterative Gates**.

### Principle: Checkpoint Continuity
- **Never restart from scratch** when progressing to higher gates.
- Every gate must serialize complete resumable state (`model`, `optimizer`, `lr_scheduler`, `scaler`, `step`/`epoch` index, replay buffer pointers).
- Progressing from Gate $N \to N+1$ must resume directly from Gate $N$'s checkpoint using `--resume_from <path>`.

---

### Gate 1: Smoke & Mechanics POC (Sanity Check)
- **Scale**:
  - Atari: 3,000 – 5,000 environment steps (~5–8 min, < $0.03).
  - CARLA: 1 epoch or 500 mini-batches (~8–10 min, < $0.04).
- **Goal**: Confirm syntax, shape compatibility, gradient flow, loss finiteness, and absence of immediate single-action collapse.
- **Pass Criteria**:
  - Loss is finite and trending down.
  - Action distribution entropy > 1.0 (no single action > 85%).
  - Zero CUDA OOM errors, VRAM headroom $\ge 15\%$.
- **Action on Failure**: Abort immediately. Zero compute budget burned.
- **Artifact**: Save `ckpt_gate1.pth`.

---

### Gate 2: Early Dynamics & Signal Validation (Convergence Check)
- **Scale**: Resumes from `ckpt_gate1.pth`.
  - Atari: Step 5,000 $\to$ 20,000 steps (~30 min, < $0.15).
  - CARLA: Epoch 1 $\to$ 5 epochs (~1.5 hours, < $0.40).
- **Goal**: Validate that the agent is learning non-trivial signal and interacting purposefully with the environment.
- **Pass Criteria**:
  - Atari: Evaluation score > 0 (or distinct rally dynamic), value prediction error dropping, reward prediction correlation positive.
  - CARLA: Held-out validation ADE < 0.45m, Lateral Error < 0.08m (8 cm), Waypoint Loss < 0.60.
- **Action on Failure**: Stop. Inspect value/policy balance or latent dynamics. Do not scale until the failure mode is understood.
- **Artifact**: Save `ckpt_gate2.pth`.

---

### Gate 3: Mid-Scale Stability & Robustness Check
- **Scale**: Resumes from `ckpt_gate2.pth`.
  - Atari: Step 20,000 $\to$ 50,000 steps (~1.5 hours, < $0.40).
  - CARLA: Epoch 5 $\to$ 20 epochs (~6 hours, ~$1.50).
- **Goal**: Check for late-stage stability, absence of policy drift or catastrophic forgetting, and multi-seed consistency.
- **Pass Criteria**:
  - Atari: Score sustained or advancing (e.g. Breakout score $\ge 10.0$), 4-way action balance maintained, search depth calibrated.
  - CARLA: Held-out validation ADE < 0.35m, Lateral Error < 0.04m (4 cm), zero collision divergence on validation routes.
- **Action on Failure**: If drift occurs, rollback to `ckpt_gate2.pth` with targeted regularization adjustments rather than restarting from step 0.
- **Artifact**: Save `ckpt_gate3.pth`.

---

### Gate 4: Full Benchmark Production & Two-Stage Archive
- **Scale**: Resumes from `ckpt_gate3.pth`.
  - Atari: Step 50,000 $\to$ 100,000 steps (literature comparison).
  - CARLA: Epoch 20 $\to$ 50 epochs (or full 220-route Bench2Drive closed-loop evaluation).
- **Goal**: Final research champions and publication-grade metrics.
- **Mandatory Finalization Pipeline**:
  1. **Stage 1 (External Archive First)**: Sync all checkpoints, telemetry CSVs, logs, and MLflow runs to `E:\MThesis_EXP`.
  2. **Stage 2 (Hugging Face Second)**: Push champion weights and rollout videos to Hugging Face.
