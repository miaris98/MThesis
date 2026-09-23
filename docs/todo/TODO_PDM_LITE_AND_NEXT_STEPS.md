# Next Steps: Full Exploitation of PDM-Lite Dataset & Advanced Policy Architecture

Following the migration to the RTX 4090 instance and the parallel co-utilization of CARLA World-on-Rails Distillation and Atari 100k MCTS, this document tracks all tried approaches, empirical outcomes, active runs, and remaining tasks.

> **Master Experiment Tracker**: For the complete 100-hypothesis experimental matrix covering dataset scaling, spatial perception, sequence modeling, multi-task losses, multimodal fusion, and closed-loop control, see [`TODO_CARLA_EXPERIMENTS.md`](file:///c:/Users/miari/Desktop/MThesis/docs/todo/TODO_CARLA_EXPERIMENTS.md).

---

## Phase 1: PDM-Lite Dataset Expansion & Ingestion (CARLA)
- [x] **Download Remaining PDM-Lite Towns**:
  - Expanded coverage across Town01, Town02, Town03, Town04, Town05, Town10, Town06, Town07 (~79 GB total).
  - Scaled route coverage from 214 routes to **~1,840+ distinct routes** (250,945 total frames across 1,730 route groups).
  - Validation split: 230,468 training frames / 20,477 validation frames.
  - *Empirical Outcome*: Completely eliminated the dataset capacity ceiling and memorization plateau observed in Town01/02-only runs.
- [x] **High-Speed Storage Migration & DataLoader Optimization**:
  - Dataset unpacked to local PCIe Gen 4 NVMe scratch SSD (`/workspace/dataset/wor_trajectories`).
  - Scaled PyTorch DataLoader to `num_workers=12`, enabled `pin_memory=True`, and `prefetch_factor=2`.
  - *Empirical Outcome*: Throughput surged from 130 samples/s (25.6 min/epoch) to **1,000–1,450 samples/s (2.6–3.7 min/epoch)** — an **~8× wall-clock speedup**.

---

## Phase 2: Architecture & Supervision Levers
- [x] **Spatial Visual Route Projection (`--route_overlay 1`)**:
  - Grounding ego-frame planned route lines directly onto front camera RGB pixels before the vision encoder.
  - *Empirical Outcome*: Slashed lateral tracking error down to <1.5 cm on training trajectories.
- [x] **Pretrained TransFuser++ RegNetY-032 Backbone**:
  - Extracted and integrated `model_0030_0.pth` (462 MB) vision encoder with frozen backbone initialization (494 frozen tensors, 108 trainable head tensors).
  - *Empirical Outcome*: Training VRAM stabilized at ~8.5 GB with zero CUDA OOM errors.
- [x] **Auxiliary Target Speed Loss (`--target_speed_loss_weight 0.2`)**:
  - 8-bin target speed auxiliary head explicitly supervising deceleration and stationary stopping.
- [x] **Aspect-Preserving Rectangular Input (`--img_size 192x512`) & Learned Tokens**:
  - Preserves roadside curb and lane geometry without square distortion.
- [x] **Progressive 4-Gate Scaling Protocol**:
  - [x] **Gate 1 (Smoke & Mechanics)**: Gradient flow, token sequence (16 vision + 4 state), loss backward verified.
  - [x] **Gate 2 (Signal Validation)**: Train ADE < 0.45m, Lat Err < 8cm confirmed.
  - [x] **Gate 3 (Stability & Robustness Milestone - Epoch 20)**:
    - **Val Lateral Error**: Dropped from 3.8 cm to **3.2 cm** (New Champion!).
    - **Val ADE**: Dropped from 0.342m to **0.249m**.
  - [ ] **Gate 4 (Full Benchmark Production - In Flight)**:
    - Actively training **Epoch 38 / 50**.
    - Epoch 37 training metrics: **ADE = 0.055m (5.5 cm)**, **Lat Err = 0.009m (9 mm)**, **Total Loss = 0.1170**.
    - ETA to Epoch 50 completion: ~11:35 local time.

---

## Phase 3: Closed-Loop Benchmarking on CARLA Leaderboard
- [x] **Full 38-Route Bench2Drive Official Leaderboard Evaluation**:
  - Evaluated 38/38 routes on `regnety032_b2d38`.
  - Identified and fixed metric classification bug where `"TickRuntime"` was treated as "unrecognised" rather than a real driving outcome ([S-047]).
  - *Empirical Outcome*: **Mean score_composed = 58.53** across all 38 routes (0 agent-fault, 0 sim-fault, 0 unrecognised).
- [ ] **Closed-Loop Evaluation of Epoch 50 Champion**:
  - Benchmark new `best_model.pth` (Epoch 20-50 champion) on the 20-route stratified evaluation subset (`data/bench2drive_subset20.txt`).
  - Extract Driving Score (DS), Route Completion (RC), and Infraction Score.
- [ ] **Two-Stage Model Synchronization**:
  - **Stage 1 (External Archive First)**: Stream final checkpoints, telemetry logs, and MLflow runs to `E:\MThesis_EXP\carla_wor_8towns_epoch50_final\`.
  - **Stage 2 (Hugging Face Second)**: Push champion model weights and rollout video summaries to Hugging Face Hub (`Miaris/mthesis-carla-wor`).

---

## Phase 4: Atari 100k Benchmark & Policy Collapse Resolution
- [x] **Investigation & Root-Cause Diagnosis of Policy Collapse (S-024 to S-047)**:
  - *Tried*: E-series parameter tuning, actor-head zero init, removing entropy loss, increasing learning rate.
  - *Finding 1 (S-043)*: `ImpalaCNNEncoder` was missing `kaiming_normal_` weight initialization; adding it broke policy invariance.
  - *Finding 2 (S-045)*: `wrap_lightzero` lacked `FireResetWrapper`, causing non-FIRE policies to hang for 108,000 steps (~2 hours) per eval episode. Fixed with protocol-compliant wrapper.
  - *Finding 3 (S-046 & S-049)*: Pure reactive on-policy PPO with GTrXL cannot achieve sample efficiency within the strict 100,000-frame budget.
- [x] **Architectural Shift: Off-Policy Model-Based MCTS Agent (`train_mcts_offpolicy.py`)**:
  - Implemented latent dynamics model (`unroll_branches`) with transition predictor, reward head, and critic head.
  - Off-policy replay buffer with GAE($\lambda=0.95$, $\gamma=0.997$) target returns.
  - Monte Carlo Tree Search (PUCT algorithm) with 8 to 50 simulations and Dirichlet noise exploration.
- [x] **S049 Benchmark Resumption from Step 38,000 (In Flight)**:
  - Resumed parallel execution across CPU cores 16–31:
    - `S049a_mcts_sim20` (seed 42)
    - `S049b_mcts_sim35` (seed 42)
    - `S049c_mcts_sim50` (seed 42)
  - *Empirical Outcome at Step 40,000 Eval*:
    - **Score: 1.00** (breaking the 0.00 collapse).
    - **Action Distribution**: NOOP 55%, RIGHT 19%, LEFT 22%, FIRE 4% (balanced 4-way exploration).
    - **Action Entropy**: **1.308** (healthy exploration).
  - *Current Status*: Actively stepping at **Step 46,000–47,000 / 100,000**.
  - ETA to 100k completion: ~11:15 local time.
- [ ] **Multi-Seed Production Run & Final HNS Score Aggregation**:
  - Once seed 42 completes, run seeds 0 and 1.
  - Calculate Human-Normalized Score (HNS) against benchmark literature (DER, SimPLe, EfficientZero).
  - Stage 1 Sync to `E:\MThesis_EXP\results_atari_100k_mcts\`.
