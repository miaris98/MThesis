# Next Steps: Full Exploitation of PDM-Lite Dataset & Advanced Policy Architecture

Following completion of the active training runs (Atari Optuna 100k on Box 1 and Qwen-30M Augmented on Box 2), the following sequenced action plan unlocks the full potential of the PDM-Lite dataset and modern transformer policy techniques.

---

## Phase 1: PDM-Lite Dataset Expansion (Box 2)
Box 2 currently has over **1,028 GB of free NVMe disk space**.
- [ ] **Download Remaining PDM-Lite Towns**:
  - Run `python scripts/setup/download_pdm_lite.py --towns Town03,Town04,Town05 --reserve-gb 40`
  - Scales route coverage from 214 routes (Town01 + Town02) to **~1,840+ distinct routes** (Town01, Town02, Town03, Town04, Town05).
  - Gives unprecedented statistical power for held-out validation and route generalization (solving Challenge 13.19).
  - *Empirical Motivation*: In the 50-epoch run, `qwen30m` plateaued at Epoch 31 (`val_loss: 0.6534`, `val_lat_err: 0.078m`) on Town01/Town02 before mild memorization began (see [`EXPERIMENT_NOTE_QWEN30M_OVERFITTING.md`](file:///c:/Users/miari/Desktop/MThesis/docs/design/EXPERIMENT_NOTE_QWEN30M_OVERFITTING.md)). Scaling $8\times$ to Town03-05 directly solves this dataset capacity ceiling.

---

## Phase 2: High-Value Architecture & Supervision Levers
- [ ] **Spatial Visual Route Projection (`--route_overlay 1`)**:
  - Render ego-frame planned route lines directly onto front camera RGB pixels before the vision encoder sees the image (`src/config/route_overlay.py`).
  - Evaluates whether grounding navigation intent in the visual coordinate frame resolves complex junction turning.
- [ ] **Auxiliary Target Speed Loss (`--target_speed_loss_weight 0.2`)**:
  - Connect the 8-bin target speed auxiliary head (`TargetSpeedHead`).
  - Explicitly supervises deceleration and stationary stopping at red traffic lights and lead vehicles.
- [ ] **Aspect-Preserving Rectangular Input (`--img_size 192x512 --crop_bottom_frac 0.25 --vision_grid 6x16`)**:
  - Matches the native TransFuser++ input aspect ratio, eliminating the 2x horizontal squash from square 256x256 resizing and preserving roadside curb/lane geometry.
- [ ] **Test CARLA-Domain UFLD ResNet-18 Vision Backbone**:
  - A/B test the ImageNet-pretrained ResNet-34 against the CARLA lane-detection pretrained UFLD backbone (`jkdxbns/autonomous-driving-carla`) via `--weights_path`.

---

## Phase 3: Closed-Loop Benchmarking on CARLA Leaderboard
- [ ] **Full 20-Route Bench2Drive Closed-Loop Comparison**:
  - Benchmark checkpoints on the 20-route stratified evaluation subset (`data/bench2drive_subset20.txt`):
    1. `cnn_baseline`
    2. `cnn_augmented` (recovery cameras)
    3. `qwen30m_augmented` (vision transformer + recovery cameras)
    4. `qwen30m_augmented_overlay` (+ spatial route projection)
  - Extract Driving Score (DS), Route Completion (RC), and Infraction Score using `scripts/eval/compare_closed_loop.py`.
- [ ] **Automated Hugging Face Sync & Video Dumps**:
  - Sync best checkpoint weights to Hugging Face Hub (`Miaris/mthesis-carla-wor`).
  - Generate front-camera video rollouts with telemetry overlays for thesis qualitative analysis (`scripts/eval/record_eval_video.py`).

---

## Phase 4: Atari Follow-Up (Box 1)
- [ ] **Inspect Optuna 100k Bayesian Study**:
  - Review top hyperparameter trials from `results/atari_gtrxl_optuna.db` via `atari_qwen/scripts/query_study.py`.
- [ ] **Train Champion GTrXL Agent**:
  - Launch full-length training (1M - 10M steps) on the winning hyperparameter configuration using `atari_qwen/scripts/run_champion_and_record.py`.
  - Record full 3-minute gameplay MP4 with live Q-value / attention visualization.
## Phase 4: Atari GTrXL / EZ2 Investigation Status & Next Steps (Box 1)
- [x] **Diagnostic & Root Cause of Policy Collapse (S-035 & S-036)**:
  - **S-035 Finding**: Policy invariance was traced directly to GTrXL's GRU skip gates (`bg_init=2.0`). At `bg=2.0`, `z = sigmoid(-2.0) ~= 0.12`, so 8 compounding gates let only tiny residual visual features through, leaving policy decisions governed almost entirely by constant input-independent tokens.
  - **S-036 Finding**: Testing `bg_init=0.0` for 15k steps confirmed that while gates start more open, the gates remain unmoved from init because sparse Atari Breakout rewards require substantially more gradient updates to break constant-action symmetry.
  - **Architectural Enhancements Implemented**:
    1. Independent per-environment epsilon-greedy sampling (fixed correlated batch noise).
    2. Exact behavior-policy logp tracking for PPO-clipped importance sampling.
    3. Multi-step GAE($\lambda=0.95$) value and advantage estimation inside trajectory windows.
    4. Small cycling replay buffer (capacity 15,000) to keep replay data fresh.
- [ ] **Decision & Next Step for Atari**:
  - Run a scaled training run (100k–300k steps) with `bg_init=0.0` or `-1.0` and 32 parallel environments so the policy has sufficient gradient iterations to learn visual feature discrimination.
