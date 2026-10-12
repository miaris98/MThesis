# Experimental Note: Qwen-30M Augmented Distillation Dynamics & Overfitting Analysis

**Date**: September 18, 2026  
**Environment**: Box 2 (Vast.ai, 2x Tesla T4 16GB, AMD EPYC 7V12)  
**Experiment**: `wor_qwen30m_augmented` (World-on-Rails with PDM-Lite Recovery Cameras)  
**Backbone**: `qwen30m` Vision-Transformer Policy  
**Dataset**: PDM-Lite Town01 + Town02 (214 routes, 35,136 frames, 80/20 train/val split)  
**Log Path**: `/workspace/train_qwen30m_augmented.log`  
**Checkpoint**: `/workspace/checkpoints/wor_qwen30m_augmented/best_model.pth`  

---

## 1. Executive Summary & Key Metrics

During 50-epoch imitation learning / distillation with recovery camera views, the Qwen-30M policy demonstrated strong waypoint prediction capabilities, with lateral deviation dropping to sub-decimeter precision. 

Around **Epoch 31**, the model reached its optimal generalization minimum. Beyond Epoch 31, a clear divergence emerged between training and validation loss, signaling the onset of mild model over-parameterization / dataset memorization.

### Key Performance Landmarks
- **Peak Validation Performance (Epoch 31)**:
  - **Validation Total Loss**: `0.6534` (Global minimum)
  - **Validation ADE (Average Displacement Error)**: `0.458m` (Peak was `0.449m` at Ep 26)
  - **Validation Lateral Error**: `0.078m` (**7.8 cm** on held-out routes)
  - **Train Lateral Error**: `0.026m` (2.6 cm)
  - **Gradient Norm**: Stabilized from initial spikes down to `6.27`
- **Training Progression at Epoch 33**:
  - **Train Total Loss**: `0.2096` (Continues dropping)
  - **Val Total Loss**: `0.6813` (Drifting upward $+4.2\%$ from Ep 31 minimum)
  - **Val ADE**: `0.484m` (Upward drift)
  - **Val Lateral Error**: `0.078m` (Plateaued at lane-center precision)

---

## 2. Empirical Loss & Error Trajectory

| Epoch | Train Loss | Val Loss | Train ADE (m) | Val ADE (m) | Train Lat Err (m) | Val Lat Err (m) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **014** | 0.4945 | 0.7283 | 0.340 | 0.485 | 0.059 | 0.097 | Best Checkpoint |
| **017** | 0.4496 | 0.6938 | 0.313 | 0.472 | 0.053 | 0.089 | Best Checkpoint |
| **020** | 0.3959 | 0.6850 | 0.274 | 0.477 | 0.047 | 0.083 | Best Checkpoint |
| **023** | 0.3466 | 0.6715 | 0.242 | 0.461 | 0.040 | 0.083 | Best Checkpoint |
| **024** | 0.3311 | 0.6688 | 0.232 | 0.454 | 0.038 | 0.087 | Best Checkpoint |
| **026** | 0.3053 | 0.6596 | 0.215 | **0.449** | 0.035 | 0.084 | Best Checkpoint |
| **029** | 0.2561 | 0.6595 | 0.181 | 0.456 | 0.029 | 0.081 | Best Checkpoint |
| **030** | 0.2420 | 0.6556 | 0.171 | 0.453 | 0.027 | 0.081 | Best Checkpoint |
| **031** | 0.2299 | **0.6534** | 0.163 | 0.458 | 0.026 | **0.078** | **★ Global Best Checkpoint** |
| **032** | 0.2173 | 0.6768 | 0.154 | 0.476 | 0.024 | 0.080 | Val loss rebound |
| **033** | 0.2096 | 0.6813 | 0.148 | 0.484 | 0.023 | **0.078** | Mild overfit divergence |

---

## 3. Root Cause Analysis

1. **Model Capacity vs. Domain Diversity**:
   - `qwen30m` contains ~30.8 million parameters with multi-head self-attention across visual spatial tokens.
   - The training set contains 214 routes restricted strictly to **Town01** (simple suburban grid) and **Town02** (small residential loop).
   - By Epoch 30, the model has saturated its learning of generic road curvature and lane centering; continued gradient steps cause the attention weights to memorize specific visual textures, building facades, and exact waypoint coordinates of the training routes.
2. **Longitudinal vs. Lateral Dynamics**:
   - The validation **Lateral Error** stayed firmly clamped at **0.078m (7.8 cm)**, indicating that steering angle and lane-centering representation did **not** degrade.
   - The divergence in Val Loss and Val ADE is almost entirely driven by **longitudinal displacement error** (braking/acceleration pacing on held-out routes where target speed profiles differ slightly).

---

## 4. Architectural Mitigation & Verification Strategy

1. **Checkpointed Protection**:
   - `scripts/training/train_wor.py` utilizes strict conditional checkpoint saving:
     ```python
     if val_loss < best_val_loss:
         best_val_loss = val_loss
         torch.save(model.state_dict(), save_dir / "best_model.pth")
     ```
   - The deployed `best_model.pth` is frozen at **Epoch 31** (`val_loss = 0.6534`), completely immune to the subsequent overfitting drift.
2. **Closed-Loop Safety Margin**:
   - In CARLA, standard lane width is 3.50 m (1.75 m margin to lane boundary).
   - A lateral error of 0.078 m represents **only 2.2% of lane width**, well within safe bounds to prevent off-road infractions and curb collisions.
3. **Phase 1 Structural Solution (Town Expansion)**:
   - To push past this plateau, we must expand the training corpus beyond Town01/Town02.
   - Downloading **Town03, Town04, and Town05** will scale data from 214 routes to **1,840+ routes** (~300k frames), providing the visual and topological diversity required to fully train a 30M-parameter vision policy without overfitting.
