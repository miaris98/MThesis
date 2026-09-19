# CARLA Autonomous Driving: Experiment Roadmap & Hypothesis Tracker

**Problem Statement**: The World-on-Rails (WoR) distillation pipeline using a frozen ResNet-34 backbone and Qwen-30M Decision Transformer policy head achieved strong offline metrics on Town01/Town02 (Epoch 34: Val Loss `0.6439`, Val ADE `0.449m`, Val Lat Err `0.075m`), but exhibits mild validation plateauing/memorization beyond Epoch 34. To scale closed-loop Driving Score (DS) and route completion on Bench2Drive, this roadmap is structured into **two distinct operational phases**:

---

## Sensor Scope & Operational Constraints

### Phase 1 Sensor Envelope (Active Priority — Strict Budget)
To maintain real-world deployment viability, low computational overhead, and edge-device feasibility, **Phase 1 strictly restricts inputs to**:
1. **RGB Cameras (Max 3)**: Center/Front (0°), Left (-60° or lateral recovery offset), Right (+60° or lateral recovery offset).
2. **Waypoints & Navigation Commands**: High-level planner commands (Turn Left, Turn Right, Go Straight, Follow Lane) and target 2D waypoint coordinates.
3. **Ego-Telemetry**: Vehicle speed $v$, heading angle $\theta$, steering angle $\delta$, acceleration $a$.
4. **Calculated / Derived Quantities (Vision & Kinematics Only)**:
   - *Geometric Transforms*: Frenet frame coordinates $(s, d)$, lateral error $e_{lat}$, longitudinal error $e_{lon}$, path curvature $\kappa = \frac{d\theta}{ds}$, tangent heading error $\Delta \theta$.
   - *Kinematic Derivatives*: Velocity vectors $\dot{w}_k$, acceleration $\ddot{w}_k$, trajectory jerk $\dddot{w}_k$.
   - *Vision-Derived Representations*: Visual route projection overlays (rendering 2D planned route onto RGB pixels), multi-frame temporal stacking ($t-1, t$), RGB optical flow, patch tokenizers, attention entropy maps, and self-supervised/supervised auxiliary monocular depth heads (predicted from RGB).
   - *Kinematic Safety Calculations*: Calculated Time-To-Collision ($\text{TTC} = \frac{d_{est}}{v}$), stopping distance bounds ($d_{stop} = \frac{v^2}{2a_{max}}$), Ackermann turning radius limits.
   - **STRICTLY EXCLUDED in Phase 1**: Physical LiDAR point clouds, 4D Radar, hardware depth sensors, external HD-map polylines, and ground-truth semantic segmentation camera streams.

### Phase 2 Sensor Envelope (Deferred — Multimodal Hardware Expansion)
Explores additional physical hardware sensors once the vision-centric Phase 1 model reaches its benchmark limits:
- Physical 32/64-beam LiDAR (range-view projection, 3D point clouds, voxelization).
- Sparse 4D Doppler Radar (penetration through heavy rain and fog).
- Hardware Active Depth Cameras (ToF / Stereo IR).
- External Vectorized HD-Map receivers and V2X telemetry.
- Direct ground-truth semantic segmentation streams.
- Multi-modal sensor dropout, dynamic gating, and hardware-redundant AEB.

---

## Current Baseline to Beat (Epoch 34 Checkpoint)
- **Model**: `wor_qwen30m_augmented` (`results/checkpoints_box2/best_model.pth`, 106 MB)
- **Validation Loss**: `0.6439` | **Validation ADE**: `0.449m` | **Validation Lateral Error**: `0.075m` (7.5 cm)
- **Training Throughput**: ~43 samples/sec on 2x Tesla T4 (~9.9 min/epoch)
- **Closed-Loop Target**: Exceed ResNet-34 CNN baseline on 20-route Bench2Drive stratified benchmark.

---

# PART 1: PHASE 1 — Vision-Centric Distillation & Control
*(Max 3 Cameras + Waypoints + Kinematic/Calculated Features)*

---

## Category 1: Dataset Scaling & Sampling Strategies (C1–C7)

### [ ] C1. 5-Town PDM-Lite Scale-Up (Town01–Town05)
- **Hypothesis**: The Town01/Town02 dataset (214 routes, ~30k frames) hits a capacity ceiling for the 27.5M parameter Qwen-30M trunk. Scaling 8x to Town01–05 (~1,840 routes, ~250k frames) will eliminate overfitting and improve generalization.
- **Method**: Run `python scripts/setup/download_pdm_lite.py --towns Town03,Town04,Town05 --reserve-gb 40` and train for 50 epochs.
- **Metric**: Validation ADE < 0.35m on held-out towns; zero val loss bounce.

### [ ] C2. High-Curvature & Intersection Oversampling
- **Hypothesis**: Straight-line cruising dominates >70% of driving data, biasing the transformer towards zero-steering priors.
- **Method**: Weight sample probabilities by $| \kappa | = | \frac{d\theta}{ds} |$ (curvature) so turns and roundabouts are sampled at 2.5x frequency.
- **Metric**: Junction collision rate reduction on Bench2Drive; Val lateral error on turns < 0.05m.

### [ ] C3. Recovery Trajectory Ratio Sweep (0%, 25%, 50%, 75%)
- **Hypothesis**: Without perturbed recovery trajectories, the policy suffers from DAgger-style distribution drift in closed-loop. Too much recovery data degrades nominal smooth driving.
- **Method**: Train 4 models with recovery frame fractions $\{0.0, 0.25, 0.50, 0.75\}$ using PDM-Lite perturbed camera views.
- **Metric**: Closed-loop Route Completion (RC) on Bench2Drive Town04.

### [ ] C4. Weather & Lighting Stratified Sampling
- **Hypothesis**: ClearNoon frames dominate the dataset, leading to severe performance drops under WetSunset and HardRain.
- **Method**: Implement stratified batching ensuring each batch contains equal representation of Clear, Rain, Sunset, and Foggy weather presets.
- **Metric**: Variance of Driving Score across all 14 CARLA weather conditions.

### [ ] C5. Dynamic Obstacle Density Stratification
- **Hypothesis**: Routes with dense traffic teach proactive braking, while empty routes teach speed maintenance. Unbalanced mixing causes indecisive creeping.
- **Method**: Tag route segments by lead-vehicle proximity (<20m) and balance batches 50/50 between open-road and convoy driving.
- **Metric**: Zero rear-end collisions; mean driving speed within 5% of target speed limit.

### [ ] C6. Stationary / Red-Light Balancing
- **Hypothesis**: Long waits at red lights inflate $v=0$ frames, causing the policy to "freeze" after stopping even after the light turns green.
- **Method**: Cap consecutive stationary frames per route group to a maximum of 10 in training.
- **Metric**: Green-light departure latency (<1.0s); zero intersection gridlock timeouts.

### [ ] C7. Synthetic Camera Translation & Yaw Jitter
- **Hypothesis**: Discrete physical recovery cameras (+0.4m, -0.4m, +/-15 deg) leave intermediate gaps. Continuous synthetic jitter fills the recovery distribution.
- **Method**: Apply random horizontal image roll and homography warps simulating $[-0.2m, +0.2m]$ lateral offset with corresponding waypoint adjustment.
- **Metric**: Val lateral error < 0.06m; recovery success rate > 95%.

---

## Category 2: Visual Perception & Input Representations (Max 3 Cameras) (C8–C15)

### [ ] C8. Aspect-Preserving Rectangular Input (192x512)
- **Hypothesis**: Squashing native CARLA 600x800 camera frames into square 256x256 distorts roadside curbs, lane lines, and vehicle aspect ratios by 2x horizontally.
- **Method**: Train with `--img_size 192x512 --crop_bottom_frac 0.25 --vision_grid 6x16` (96 visual tokens).
- **Metric**: Lane-keeping out-of-lane infraction reduction by >50%.

### [ ] C9. Spatial Visual Route Overlay (`--route_overlay 1`)
- **Hypothesis**: High-level commands ("Turn Left") provide ambiguous geometric grounding. Projecting the 2D planned route line onto RGB pixels provides direct pixel-level guidance.
- **Method**: Enable ego-frame route projection onto camera images before the vision backbone (`src/config/route_overlay.py`).
- **Metric**: Route Completion (RC) on complex Town03/04 multi-lane intersections.

### [ ] C10. CARLA-Domain UFLD Lane-Pretrained Backbone
- **Hypothesis**: ImageNet weights specialize in object classification (dogs, cats), whereas CARLA lane-detection features (UFLD) directly encode road surface geometry.
- **Method**: Replace ImageNet ResNet-34 with the CARLA-pretrained UFLD ResNet-18/34 backbone (`jkdxbns/autonomous-driving-carla`).
- **Metric**: Convergence rate (2x faster); initial epoch Val ADE < 0.8m.

### [ ] C11. Multi-Camera Surround Fusion (Left -60°, Center 0°, Right +60°)
- **Hypothesis**: A single front camera has a 90-degree FOV blind spot to cross-traffic at unsignalized intersections. Exactly 3 cameras cover 210° FOV.
- **Method**: Encode Left (-60 deg), Center (0 deg), and Right (+60 deg) images with shared ResNet, concatenating tokens with camera-angle positional embeddings.
- **Metric**: Crossing-vehicle collision rate on Bench2Drive reduced to zero.

### [ ] C12. Unfreezing Top Residual Stage of Vision Backbone (Stage 4)
- **Hypothesis**: Frozen ImageNet features cannot adapt to CARLA's synthetic rendering textures. Unfreezing Stage 4 with 0.1x learning rate allows domain adaptation without catastrophic forgetting.
- **Method**: Set `backbone.layer4.requires_grad = True` with `lr = 1e-4` while keeping layers 1–3 frozen.
- **Metric**: Validation loss reduction below 0.60.

### [ ] C13. Photometric & Contrast Jitter Augmentation
- **Hypothesis**: Direct sunlight glare and night headlight shadows wash out features.
- **Method**: Apply ColorJitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1) and RandomAutoContrast at $p=0.5$.
- **Metric**: Performance parity between Day and Night/Dusk routes.

### [ ] C14. Multi-Frame Temporal Visual Stacking ($t-1, t$)
- **Hypothesis**: Single-frame input cannot estimate vehicle velocities directly from pixels, relying solely on scalar speed telemetry.
- **Method**: Stack the current frame and previous frame ($t-1$) as a 6-channel tensor before the visual backbone.
- **Metric**: Velocity tracking accuracy; reduced false-braking behind moving lead cars.

### [ ] C15. ViT Patch Tokenizer (14x14 Patches)
- **Hypothesis**: CNN pooling loses precise pixel-to-token spatial coordinates. Direct patch embedding retains exact spatial layout.
- **Method**: Use `PatchTokenizer(in_channels=3, embed_dim=256, patch_size=14)` on 224x448 input (512 tokens).
- **Metric**: Visual token attention entropy; spatial alignment of saliency maps with pedestrians.

---

## Category 3: Transformer Architecture & Sequence Modeling (C16–C23)

### [ ] C16. Model Capacity Scaling Curve (10M -> 30M -> 100M)
- **Hypothesis**: On the expanded 5-Town dataset, Qwen-100M will outperform Qwen-30M, whereas on 2 towns it overfit.
- **Method**: Train Qwen-10M, Qwen-30M, and Qwen-100M policy heads on Town01–05 under identical learning rate schedules.
- **Metric**: Pareto curve of Val ADE vs Parameter Count; inference latency on RTX A4000.

### [ ] C17. Rotary Position Embeddings (RoPE) for 2D Grid Tokens
- **Hypothesis**: Standard 1D learnable positional embeddings do not preserve 2D horizontal/vertical spatial geometry of camera feature maps.
- **Method**: Apply 2D axial Rotary Position Embeddings (RoPE) to visual grid tokens.
- **Metric**: Relative position attention fidelity; lateral error on sharp S-curves.

### [ ] C18. Cross-Attention Query Bottleneck (Perceiver Style)
- **Hypothesis**: In full self-attention, 64 visual tokens dominate the attention matrix, diluting the 4 decision tokens.
- **Method**: Use a Perceiver-style cross-attention layer where `[POLICY, STATE]` tokens query visual tokens as keys/values, followed by self-attention only among decision tokens.
- **Metric**: 30% reduction in compute FLOPs; higher attention weight on lead vehicles.

### [ ] C19. Causal Temporal Transformer over Trajectory History ($H=4$)
- **Hypothesis**: Markovian (single-step) policies exhibit oscillatory steering. Conditioning on the past 4 ego-waypoints smooths trajectories.
- **Method**: Feed past ego states $[s_{t-3}, s_{t-2}, s_{t-1}, s_t]$ into a causal transformer block.
- **Metric**: Steering jerk metric $\frac{1}{T} \sum |\ddot{\delta}|$; passenger comfort score.

### [ ] C20. Stochastic Token Dropping (Patch Dropout 20%)
- **Hypothesis**: The policy over-relies on specific background landmarks (buildings, trees) instead of road geometry.
- **Method**: Randomly mask out 20% of visual tokens during training.
- **Metric**: Generalization gap between seen Town01 and unseen Town04 reduced by >40%.

### [ ] C21. SwiGLU Feed-Forward Networks vs Standard GELU MLP
- **Hypothesis**: SwiGLU activations provide superior gating for continuous geometric coordinates compared to standard GELU.
- **Method**: Replace standard two-layer GELU MLP in Qwen blocks with SwiGLU FFN (`w_down(silu(w_gate(x)) * w_up(x))`).
- **Metric**: Training convergence rate (epochs to reach Val ADE < 0.5m).

### [ ] C22. Attention Entropy Regularization
- **Hypothesis**: Self-attention heads often collapse onto the `[STATE]` token or the ego-hood area.
- **Method**: Add an entropy loss on the attention matrix $L_{attn} = - \sum A \log A$ to encourage wide visual scanning.
- **Metric**: Number of active attention heads attending to pedestrians and traffic lights.

### [ ] C23. Bidirectional vs Causal Self-Attention
- **Hypothesis**: For a single-timestep observation, visual tokens have no causal ordering. Bidirectional self-attention extracts richer scene context than causal masking.
- **Method**: A/B test full bidirectional self-attention vs causal triangular masking on the visual+state sequence.
- **Metric**: Validation loss and waypoint FDE.

---

## Category 4: Multi-Task Supervision & Auxiliary Heads (Calculated & Vision-Derived) (C24–C31)

### [ ] C24. Auxiliary Target Speed Classification Head
- **Hypothesis**: Direct waypoint regression often averages out stopping vs driving. An explicit target speed head forces decisive stopping.
- **Method**: Add 8-bin discrete speed head (`TargetSpeedHead`) with `--target_speed_loss_weight 0.2`.
- **Metric**: Red-light infraction rate on Bench2Drive reduced to zero.

### [ ] C25. Auxiliary Traffic Light State Prediction Head
- **Hypothesis**: Grounding visual tokens in traffic light semantics prevents reliance on lead-car behavior at lights.
- **Method**: Supervise an auxiliary 4-class classification head (Red, Yellow, Green, None) from CARLA ground truth.
- **Metric**: F1 score on traffic light classification; zero red-light infractions.

### [ ] C26. Auxiliary Stop Sign Distance Regression
- **Hypothesis**: Stop signs require a full stop followed by yielding. Waypoints alone often roll through without a complete stop.
- **Method**: Supervise a continuous distance-to-stop-line head with smooth L1 loss.
- **Metric**: Stop sign compliance score on Town03 intersections.

### [ ] C27. Waypoint Velocity, Acceleration & Jerk Derivatives
- **Hypothesis**: Pure position waypoints $w_k = (x_k, y_k)$ ignore temporal smoothness. Supervising $\dot{w}_k$ and $\ddot{w}_k$ enforces dynamically feasible trajectories.
- **Method**: Add $L_{deriv} = \| \Delta w_k - v_{gt} \|_1 + \| \Delta^2 w_k - a_{gt} \|_1$.
- **Metric**: Steering and acceleration smoothness metrics; zero tire-slip events.

### [ ] C28. Leading Vehicle Distance & Relative Speed Head (from RGB)
- **Hypothesis**: Tailgating occurs when depth perception is imprecise from single RGB.
- **Method**: Predict distance $d_{lead}$ and relative velocity $\Delta v_{lead}$ to the closest forward actor within 40m using visual features.
- **Metric**: Minimum time-to-collision (TTC > 2.5s at all times).

### [ ] C29. Ego-Vehicle Heading Angle Loss ($\Delta \theta$)
- **Hypothesis**: Errors in predicted heading accumulate rapidly over multi-waypoint rollouts.
- **Method**: Supervise tangent angle $\theta_k = \arctan2(y_{k+1}-y_k, x_{k+1}-x_k)$ with cosine loss $1 - \cos(\theta_{pred} - \theta_{gt})$.
- **Metric**: Heading error at 5m lookahead < 1.5 degrees.

### [ ] C30. Multi-Horizon Waypoint Prediction ($K=5$ vs $K=10$ vs $K=20$)
- **Hypothesis**: A short horizon ($K=5$) causes myopic cornering; an excessively long horizon ($K=20$) has high uncertainty.
- **Method**: Sweep prediction horizon $K \in \{5, 10, 15, 20\}$ at 0.5s intervals.
- **Metric**: Optimal trade-off between near-field lateral tracking and far-field curve anticipation.

### [ ] C31. Monocular Depth Estimation Auxiliary Head (Predicted from RGB)
- **Hypothesis**: Pure RGB models struggle with absolute metric distance. Supervising an auxiliary depth prediction head on visual tokens forces the CNN/ViT to encode true 3D spatial geometry.
- **Method**: Supervise an auxiliary depth regression head ($1/d$) on visual tokens using ground truth depth maps during training only (zero depth sensor needed at test time).
- **Metric**: Mean absolute depth error < 0.5m on lead vehicles within 25m; zero test-time sensor overhead.

---

## Category 5: Loss Function Formulations & Geometric Constraints (C32–C38)

### [ ] C32. Curvature-Weighted Waypoint Loss
- **Hypothesis**: Standard L1 loss treats a 0.2m error on a straight road identically to a 0.2m error on a tight 90-degree turn, but turn errors cause curb strikes.
- **Method**: Weight waypoint loss by $w_i = 1.0 + 3.0 \cdot |\kappa_i|$.
- **Metric**: Curb and barrier collision rate on Bench2Drive reduced to zero.

### [ ] C33. Asymmetric Lateral vs Longitudinal Loss Decomposition (Frenet)
- **Hypothesis**: Longitudinal errors (arriving 0.1s early/late) are safe; lateral errors (drifting 0.3m into oncoming traffic) cause fatal crashes.
- **Method**: Project errors into the Frenet frame and set $L_{total} = 5.0 \cdot L_{lateral} + 1.0 \cdot L_{longitudinal}$.
- **Metric**: Validation lateral error < 0.04m (4 cm).

### [ ] C34. Frenet Frame Arc-Length Decomposition Loss
- **Hypothesis**: Cartesian $(x, y)$ losses couple steering and throttle errors. Frenet coordinates $(s, d)$ decouple speed from lane-centering.
- **Method**: Supervise progress along centerline $s$ and orthogonal displacement $d$.
- **Metric**: Lane-centering metric; zero lane-straddling infractions.

### [ ] C35. Smooth L1 (Huber) vs MSE vs Log-Cosh Loss
- **Hypothesis**: MSE squares large outlier errors (e.g. sharp evasive maneuvers), destabilizing training gradients.
- **Method**: A/B test MSE ($\beta=\infty$), Smooth L1 ($\beta=1.0$), and Log-Cosh loss.
- **Metric**: Gradient norm stability; outlier waypoint error percentiles (99th percentile).

### [ ] C36. Kinematic Feasibility Penalty (Ackermann Limits)
- **Hypothesis**: The neural network can predict geometrically sharp waypoints that the vehicle's physics cannot follow.
- **Method**: Penalize predicted trajectory curvature exceeding maximum vehicle steering angle: $L_{feas} = \max(0, |\kappa| - \kappa_{max})^2$.
- **Metric**: Controller tracking error (difference between predicted waypoint and actual vehicle execution).

### [ ] C37. Safe Braking Margin Hinge Loss
- **Hypothesis**: When approaching stopped traffic, under-predicting distance is catastrophic.
- **Method**: Add asymmetric hinge loss: penalize under-braking 3x more heavily than over-braking.
- **Metric**: Zero collisions with stationary lead vehicles at traffic lights.

### [ ] C38. Minimum Jerk Trajectory Regularizer
- **Hypothesis**: High-frequency waypoint jitter creates rapid steering oscillations that wear vehicle physics in simulation.
- **Method**: Penalize third derivative: $L_{jerk} = \sum \| w_{k+2} - 2w_{k+1} + w_k \|^2$.
- **Metric**: Trajectory smoothness index; passenger comfort metric.

---

## Category 6: Optimization, Regularization & Training Schedules (C39–C44)

### [ ] C39. Cosine Annealing with Warm Restarts (SGDR)
- **Hypothesis**: Standard monotonic cosine decay gets trapped in sub-optimal local minima for complex multi-task heads.
- **Method**: Implement SGDR with $T_0 = 15$ epochs, $T_{mult} = 2$, decaying max LR from $3e-4$ to $5e-5$.
- **Metric**: Validation loss recovery and escape from plateau at Epoch 30.

### [ ] C40. Stochastic Weight Averaging (SWA)
- **Hypothesis**: A single checkpoint represents one point on the loss surface. Averaging checkpoints across late epochs finds flatter, more robust minima.
- **Method**: Apply SWA over checkpoints from Epochs 30 to 50 with learning rate $1e-4$.
- **Metric**: Generalization on unseen weather routes on Bench2Drive.

### [ ] C41. Layer-wise Learning Rate Decay (LLRD)
- **Hypothesis**: Lower layers of the transformer learn general spatial features; higher layers learn specific waypoint coordinates.
- **Method**: Apply decay factor $\alpha = 0.85$ per layer from top to bottom.
- **Metric**: Preserved representation diversity in early transformer blocks.

### [ ] C42. Weight Decay Regularization on Non-Norm Weights
- **Hypothesis**: High weight norms in the policy projection head lead to overconfident waypoint extrapolation.
- **Method**: Test weight decay values $\lambda \in \{0.001, 0.01, 0.05, 0.1\}$, strictly excluding LayerNorm, bias, and gate parameters.
- **Metric**: Overfitting gap ($L_{val} - L_{train}$) narrowed by >50%.

### [ ] C43. Large Batch Size vs Gradient Accumulation (Batch 32 vs 64 vs 128)
- **Hypothesis**: Larger batch sizes provide smoother gradient estimates across diverse driving scenarios.
- **Method**: Use gradient accumulation steps = 2 and 4 to simulate batch sizes 64 and 128 on RTX A4000.
- **Metric**: Gradient variance reduction; training throughput (samples/sec).

### [ ] C44. Mixed Precision Numerical Stability Audit (BFloat16 vs Float16)
- **Hypothesis**: Float16 underflows on small waypoint gradients. BFloat16 preserves dynamic range without requiring gradient scaler adjustments.
- **Method**: Compare training curves and gradient norms under native BFloat16 vs Float16 AMP.
- **Metric**: Zero gradient scale underflow events; identical loss convergence to Float32.

---

## Category 7: Control & Trajectory Execution (Vision & Kinematics) (C45–C48)

### [ ] C45. Longitudinal & Lateral PID Controller Optimization
- **Hypothesis**: Waypoint prediction is accurate, but default controller gains ($K_p, K_i, K_d$) cause corner-cutting or sluggish acceleration.
- **Method**: Grid search lateral PID gains ($K_p \in [0.8, 1.5], K_d \in [0.1, 0.5]$) and longitudinal PID on closed-loop routes.
- **Metric**: Lateral tracking error between predicted waypoints and executed vehicle path < 0.05m.

### [ ] C46. Model Predictive Path Integral (MPPI) Local Smoothing
- **Hypothesis**: Direct PID execution follows noisy instantaneous waypoints. MPPI optimizes control inputs over a short horizon against vehicle kinematics.
- **Method**: Wrap model outputs with an MPPI controller sampling 500 candidate rollouts over the predicted waypoints.
- **Metric**: Closed-loop trajectory smoothness and zero spin-outs on wet asphalt.

### [ ] C47. Velocity-Adaptive Lookahead Horizon ($L = k \cdot v$)
- **Hypothesis**: Fixed lookahead distance (e.g. 5m) is too short at 40 km/h (causes oscillation) and too long at 5 km/h (causes corner-cutting).
- **Method**: Set lookahead point $L(v) = \max(3.0, k \cdot v)$ with $k = 0.5s$.
- **Metric**: Turning stability at junctions; zero curb mountings during tight turns.

### [ ] C48. Vision-Kinematic Autonomous Emergency Braking (AEB) Fallback
- **Hypothesis**: Neural network edge-case failure should not result in a high-speed collision.
- **Method**: Implement a deterministic safety supervisor using predicted obstacle distance $d_{est}$ and ego-speed $v$ to compute $\text{TTC} = \frac{d_{est}}{v}$. If $\text{TTC} < 1.2s$ or $d < \frac{v^2}{2a_{max}} + 2m$, override policy with maximum emergency braking.
- **Metric**: Catastrophic collision rate reduced to absolute 0.0% across all Bench2Drive trials without external radar/depth hardware.

---

## Category 8: Closed-Loop Bench2Drive Evaluation & Failure Analysis (C49–C51)

### [ ] C49. 20-Route Stratified Bench2Drive Benchmark
- **Hypothesis**: Full offline validation cannot substitute for closed-loop interaction.
- **Method**: Evaluate champions on `data/bench2drive_subset20.txt` (10 Town01 routes, 10 Town04 routes) with `scripts/eval/compare_closed_loop.py`.
- **Metric**: Driving Score (DS), Route Completion (RC), and Infraction Score.

### [ ] C50. Weather-Stressed Closed-Loop Evaluation
- **Hypothesis**: Models trained without heavy weather augmentation fail catastrophically under WetSunset (reflections) and HardRain (visibility).
- **Method**: Run Bench2Drive evaluations under extreme weather suites.
- **Metric**: Performance degradation ratio: $\frac{DS_{extreme}}{DS_{clear}} > 0.85$.

### [ ] C51. Automated Failure Mode Categorization (`classify_failures.py`)
- **Hypothesis**: Aggregate Driving Score obscures whether failure was caused by red lights, pedestrian jaywalkers, or lane drifting.
- **Method**: Run `scripts/eval/classify_failures.py` on all evaluation logs to categorize errors into: (a) Traffic Light, (b) Vehicle Collision, (c) Pedestrian Collision, (d) Off-road, (e) Timeout.
- **Metric**: Automated failure Pareto chart for every trained checkpoint.

---

## Category 9: Camera-Only Bird's Eye View (BEV) & Spatial Geometry (C52–C55)

### [ ] C52. Multi-Camera (Left, Center, Right) BEV Feature Transformation
- **Hypothesis**: Perspective camera space suffers from scale distortion (distant objects appear tiny). Transforming 3-camera image features into a top-down metric BEV grid provides invariant spatial reasoning purely from vision.
- **Method**: Project 2D multi-scale features from Left (-60°), Center (0°), and Right (+60°) cameras into a 32x32 metric BEV grid (0.5m/cell) using cross-attention (BEVFormer style) without LiDAR.
- **Metric**: Free-space occupancy prediction accuracy; zero off-road infractions.

### [ ] C53. Camera-to-Ego Inverse Perspective Mapping (IPM) Feature Pooling
- **Hypothesis**: Flat road assumptions hold well for highway and straight urban routes. IPM projection of road surface pixels provides fast, lightweight top-down feature alignment.
- **Method**: Apply calibrated homography IPM to front camera features, pooling road tokens into an ego-centric grid.
- **Metric**: Lane boundary alignment error; 2x faster than full cross-attention BEV.

### [ ] C54. Visual Horizon Elevation Invariance (Pitch/Roll Compensation from Ego-IMU)
- **Hypothesis**: Accelerating and braking causes vehicle pitch, shifting the visual horizon up/down and corrupting depth estimation.
- **Method**: Use ego-IMU pitch and roll angles to dynamically adjust camera projection matrices before tokenization.
- **Metric**: Waypoint stability during hard acceleration and braking transitions.

### [ ] C55. Ego-Hood Visual Masking & Perspective Normalization
- **Hypothesis**: The bottom 20% of front camera images contains the vehicle's own hood, which wastes visual tokens and varies across vehicle models.
- **Method**: Crop or statically mask the ego-hood area (`--crop_bottom_frac 0.20`), reallocating tokens to the distant road horizon.
- **Metric**: 25% reduction in visual tokens with zero loss in near-field tracking.

---

## Category 10: Advanced Decision Transformer & Sequence Dynamics (C56–C62)

### [ ] C56. Return-to-Go (RTG) Conditioning
- **Hypothesis**: In classical Decision Transformers, conditioning on high target returns biases generation toward optimal behaviors.
- **Method**: Prepend an RTG token representing target driving score ($R=100.0$) and zero infractions ($I=0$) to the input sequence.
- **Metric**: Driving Score improvement when conditioning on $R=100$ vs $R=50$.

### [ ] C57. In-Context Demonstration Prompting
- **Hypothesis**: Large sequence models can perform in-context learning. Providing 2 successful recovery trajectories in the context window guides the policy out of rare edge cases.
- **Method**: Prepend 2 demonstration sequences from the expert dataset as prefix tokens before the active observation.
- **Metric**: Recovery success rate on sharp off-center initializations.

### [ ] C58. Autoregressive State-Action-Reward Sequence Modeling
- **Hypothesis**: Predicting the entire trajectory autoregressively token-by-token captures multi-modal action distributions better than parallel single-step regression.
- **Method**: Formulate trajectory generation as tokenized autoregression over discrete waypoint codebooks.
- **Metric**: Multi-modal trajectory diversity at branching highway forks.

### [ ] C59. Recurrent Memory Transformer (RMT) for Kilometer-Scale Navigation
- **Hypothesis**: Standard transformers have a finite context window ($<100$ tokens). Passing recurrent memory tokens across time preserves long-term route intent over kilometer-long drives.
- **Method**: Pass 4 recurrent memory tokens from step $t-1$ to step $t$ through the transformer blocks.
- **Metric**: Route Completion (RC) on long Bench2Drive routes (>1,000m).

### [ ] C60. FlashAttention-2 Integration for Multi-Camera Sequences
- **Hypothesis**: Processing 3 cameras simultaneously triples visual tokens (up to 288 tokens). Standard PyTorch attention scales quadratically in memory.
- **Method**: Integrate FlashAttention-2 kernels to process 3-camera visual sequences with 3x lower VRAM and 2.5x speedup.
- **Metric**: Memory footprint on RTX A4000; training throughput (samples/sec).

### [ ] C61. Mamba / State Space Model (SSM) Backbone Alternative
- **Hypothesis**: Transformers require $O(N^2)$ attention computations, whereas State Space Models (Mamba-2) provide linear $O(N)$ scaling and constant-time recurrent inference.
- **Method**: Replace the Qwen transformer trunk with a 24-layer Mamba-2 block.
- **Metric**: Inference latency at 50Hz; validation ADE parity with Qwen-30M.

### [ ] C62. Multi-Scale Temporal Horizon Queries
- **Hypothesis**: Immediate steering (0.5s) and tactical lane-following (3.0s) require different spatial features. A single query token forces feature entanglement.
- **Method**: Use 3 distinct query tokens: `[QUERY_NEAR]` (0–1s), `[QUERY_MID]` (1–3s), and `[QUERY_FAR]` (3–5s).
- **Metric**: Near-field lateral precision and far-field curvature anticipation.

---

## Category 11: Safety, Uncertainty & Out-Of-Distribution (OOD) Detection (C63–C69)

### [ ] C63. Deep Ensemble Variance for OOD Detection
- **Hypothesis**: When encountering unseen scenarios (unusual road geometry, glitchy textures), individual networks make confident but erratic errors.
- **Method**: Train a 3-seed ensemble; compute epistemic variance across predicted trajectories $\sigma_{ens}^2 = \frac{1}{M}\sum (w_i - \bar{w})^2$.
- **Metric**: Correlation between ensemble variance and imminent collisions; trigger safety stop if $\sigma_{ens} > 0.3m$.

### [ ] C64. Evidential Deep Learning / Gaussian Mixture Waypoints
- **Hypothesis**: Standard L1 loss provides no measure of confidence. Evidential deep learning outputs parameters of an evidential distribution to quantify uncertainty without ensembles.
- **Method**: Predict both mean waypoints $\mu_k$ and epistemic variance $\sigma_k^2$ via negative log-likelihood loss.
- **Metric**: Precision-recall of uncertainty scores on out-of-distribution weather conditions.

### [ ] C65. Conformal Prediction Bounds on Trajectories
- **Hypothesis**: Safety-critical systems require mathematical guarantees. Conformal prediction constructs distribution-free prediction regions with guaranteed coverage.
- **Method**: Compute non-conformity scores on validation routes to form 95% confidence safety tubes around predicted waypoints.
- **Metric**: 95% empirical coverage guarantee on test routes; zero unexpected trajectory deviations.

### [ ] C66. Energy-Based Visual OOD Sample Rejection
- **Hypothesis**: The model should recognize when it does not know what it is seeing (e.g. camera corruption, untextured meshes).
- **Method**: Train an energy-based score $E(x) = -T \cdot \log \sum e^{f_i(x)/T}$ on visual tokens to detect OOD inputs.
- **Metric**: Detection AUC > 0.95 on corrupted camera inputs; trigger graceful emergency stop.

### [ ] C67. Adversarial Perturbation Robustness Audit
- **Hypothesis**: Neural policies can be brittle to minor input noise (e.g. camera compression artifacts, raindrops on lens).
- **Method**: Evaluate policy robustness under Fast Gradient Sign Method (FGSM) and random patch noise on front camera RGB.
- **Metric**: Robustness degradation curve; maximum allowable noise $\epsilon$ before lane departure.

### [ ] C68. Shadow & Asphalt Glare Filtering
- **Hypothesis**: Harsh overhead shadows cast by bridges and trees cause sudden false-positive obstacle detections and phantom braking.
- **Method**: Implement local contrast normalization and high-pass filtering on visual tokens.
- **Metric**: False-braking rate under tree-lined boulevards and underpasses reduced to zero.

### [ ] C69. Automatic Reversal & Recovery Mode (State Machine Fallback)
- **Hypothesis**: If the vehicle collides lightly or gets stuck on a curb, forward-only policies stay stuck indefinitely.
- **Method**: Implement an automatic state-machine fallback that detects zero progress for >3s, shifts into reverse, unwinds steering, and backs up 3m.
- **Metric**: Route completion rescue rate on stuck scenarios.

---

## Category 12: Advanced Data Synthesis & Generative Augmentation (C70–C75)

### [ ] C70. Diffusion-Based Weather & Lighting Synthesis (ControlNet on RGB)
- **Hypothesis**: CARLA weather presets are discrete. Synthesizing photorealistic rain streaks, puddles, and dusk glare via ControlNet expands visual diversity infinitely.
- **Method**: Augment ClearNoon training frames with ControlNet-conditioned synthetic weather variations.
- **Metric**: Driving score improvement on unseen extreme weather conditions.

### [ ] C71. Latent World Model Imagination (Dreamer Style)
- **Hypothesis**: Simulating millions of environment steps in CARLA is slow (real-time bound). A latent world model can imagine rollouts in VRAM at 1,000 FPS.
- **Method**: Train a recurrent world model on latent tokens $z_t$ and perform policy rollouts entirely in imagination.
- **Metric**: Sample efficiency; policy improvement using imagined counterfactuals.

### [ ] C72. Counterfactual Collision Inversion
- **Hypothesis**: Failed routes contain valuable information about failure boundaries. Inverting failure states into successful evasive maneuvers teaches boundary limits.
- **Method**: Take near-miss collision frames from evaluation logs and re-label with expert evasive steering trajectories.
- **Metric**: Evasive maneuver success rate when lead vehicle brakes abruptly.

### [ ] C73. Closed-Loop DAgger Self-Play
- **Hypothesis**: Offline datasets cannot cover all states visited by the learned policy. Interactive DAgger relabels the policy's own visited states.
- **Method**: Roll out the policy in simulation, record visited states, query PDM-Lite for optimal actions, and aggregate into training dataset.
- **Metric**: Closed-loop Route Completion (RC) improvement from 70% to 90%+.

### [ ] C74. Dynamic Actor Rendering Injection
- **Hypothesis**: Empty routes fail to teach obstacle avoidance, while re-simulating traffic is expensive.
- **Method**: Synthetically paste 2D vehicle and pedestrian bounding boxes with depth-consistent blending into empty route frames.
- **Metric**: Vehicle and pedestrian collision rate reduction.

### [ ] C75. Multi-Speed Trajectory Retiming
- **Hypothesis**: Policies often overfit to the exact speed profile of the expert. Decoupling spatial geometry from speed expands data diversity.
- **Method**: Resample existing trajectories at 0.7x, 1.0x, and 1.3x speed profiles with corresponding waypoint spacing adjustments.
- **Metric**: Adaptability to dynamic speed limits (30 vs 60 vs 90 km/h).

---

## Category 13: Traffic Rules, Semantics & Complex Interactions (Vision-Only) (C76–C82)

### [ ] C76. Priority-to-the-Right & Uncontrolled Junction Negotiation
- **Hypothesis**: Uncontrolled intersections in Town03 cause severe T-bone collisions because no traffic light governs right-of-way.
- **Method**: Add an auxiliary classification head detecting uncontrolled intersections to enforce mandatory deceleration and cross-traffic checking.
- **Metric**: Collision rate at unsignalized junctions reduced by >80%.

### [ ] C77. Roundabout Insertion & Merging Gap Acceptance
- **Hypothesis**: Roundabouts require yield-on-entry and continuous gap estimation, which standard waypoint regression misjudges.
- **Method**: Supervise a binary "safe-to-merge" classification head based on circulating traffic distance and velocity predicted from camera tokens.
- **Metric**: Successful roundabout negotiation rate on Town03 without collisions or timeouts.

### [ ] C78. Overtaking & Lane-Change Intention Head
- **Hypothesis**: When blocked by a double-parked delivery truck, pure lane-keeping policies wait indefinitely.
- **Method**: Predict target lane offset relative to current lane ($-1, 0, +1$) and generate lane-change waypoints when blocked.
- **Metric**: Blocked-route resolution rate; zero permanent gridlock failures.

### [ ] C79. Pedestrian Crossing Intention Prediction
- **Hypothesis**: Pedestrians standing on the sidewalk should not trigger emergency braking unless they step into the crosswalk.
- **Method**: Supervise pedestrian motion vectors ($\dot{x}_{ped}, \dot{y}_{ped}$) from RGB to distinguish stationary bystanders from crossing pedestrians.
- **Metric**: Elimination of phantom braking behind sidewalk pedestrians while maintaining 100% stop rate for active crossers.

### [ ] C80. Emergency Vehicle Detection & Pull-Over Behavior
- **Hypothesis**: Emergency vehicles with flashing lights have absolute right-of-way.
- **Method**: Train a detector for emergency vehicle strobe lights from RGB and trigger an evasive right-shoulder pull-over maneuver.
- **Metric**: Emergency vehicle yield compliance on Bench2Drive.

### [ ] C81. Speed-Limit Sign OCR / Visual Classification Head
- **Hypothesis**: Relying on map-based speed limits fails in temporary construction zones or dynamic speed limit changes.
- **Method**: Supervise a classification head predicting road speed limits (30, 60, 90 km/h) directly from front camera tokens.
- **Metric**: Speeding infraction score reduced to absolute 0.0%.

### [ ] C82. Creep-and-Peek Blind Intersection Maneuver
- **Hypothesis**: At blind T-junctions occluded by buildings, stopping at the stop line does not provide visibility of oncoming traffic.
- **Method**: Implement a two-stage stopping controller: full stop at stop line, followed by low-speed creeping (0.8 m/s) until cross-camera (Left/Right) FOV is clear.
- **Metric**: Zero blind-junction pull-out collisions.

---

## Category 14: Model Compression, Distillation & Low-Latency Execution (C83–C87)

### [ ] C83. Teacher-to-Student Distillation into Qwen-30M Vision Trunk
- **Hypothesis**: Heavy vision models or multi-modal teachers achieve high driving scores. Distilling their intermediate representations into a compact 3-camera Qwen-30M transfers teacher competence without hardware overhead.
- **Method**: Minimize KL divergence between student and teacher waypoint distributions, plus MSE on latent feature maps.
- **Metric**: Student matches >95% of teacher Driving Score with 4x lower latency.

### [ ] C84. Post-Training Quantization (INT8 / FP8 TensorRT)
- **Hypothesis**: 30M parameter FP32 inference takes ~25ms on RTX A4000. INT8 quantization reduces latency to <8ms, enabling high-frequency control.
- **Method**: Apply TensorRT INT8 calibration on validation routes.
- **Metric**: Inference throughput (>100 FPS); zero degradation in Val ADE (<0.01m difference).

### [ ] C85. Structured Pruning of Transformer FFNs (30% Sparsity)
- **Hypothesis**: Large feed-forward layers contain significant redundant capacity for simple driving tasks.
- **Method**: Apply Wanda structured pruning to prune 30% of MLP intermediate neurons followed by 5 epochs of fine-tuning.
- **Metric**: Model size reduced from 106 MB to 75 MB; identical driving performance.

### [ ] C86. Non-Autoregressive Parallel Waypoint Decoding
- **Hypothesis**: Autoregressive decoding adds latency per waypoint ($K \times$ forward passes). Non-autoregressive parallel decoding outputs all $K$ waypoints in 1 pass.
- **Method**: Use a parallel linear projection head over the policy representation token.
- **Metric**: 1-pass forward latency < 5ms.

### [ ] C87. MobileNetV4 / EfficientNet-B0 Backbone Replacement
- **Hypothesis**: ResNet-34 is computationally heavy for edge deployment. MobileNetV4 provides comparable representation at a fraction of the FLOPs.
- **Method**: A/B test MobileNetV4-Conv-Small against ResNet-34.
- **Metric**: FPS improvement on embedded hardware (e.g. Jetson Orin); Driving Score retention.

---

## Category 15: Hybrid RL & Imitation Refinement (C88–C92)

### [ ] C88. Implicit Q-Learning (IQL) Offline RL Refinement
- **Hypothesis**: Pure imitation learning mimics sub-optimal human/expert mistakes. Offline RL with IQL extracts the best behaviors from suboptimal datasets.
- **Method**: Train an IQL value function on PDM-Lite trajectories and fine-tune the actor head with advantage-weighted regression.
- **Metric**: Infraction score reduction compared to pure behavioral cloning.

### [ ] C89. Conservative Q-Learning (CQL) on Diverse Trajectories
- **Hypothesis**: Offline Q-learning overestimates values on unseen state-action pairs. CQL penalizes out-of-distribution actions to maintain safety.
- **Method**: Apply CQL regularizer on the Q-head over the PDM-Lite dataset.
- **Metric**: Robustness on out-of-distribution initializations.

### [ ] C90. Interactive DAgger with PDM-Lite Oracle
- **Hypothesis**: Offline datasets cannot cover all states visited by the learned policy. Interactive DAgger relabels the policy's own visited states.
- **Method**: Roll out the policy in simulation, record visited states, query PDM-Lite for optimal actions, and aggregate into training dataset.
- **Metric**: Closed-loop Route Completion (RC) improvement from 70% to 90%+.

### [ ] C91. Advantage-Weighted Regression (AWR)
- **Hypothesis**: Weighting all training transitions equally incorporates low-quality recovery data. Weighting by advantage prioritizes high-reward maneuvers.
- **Method**: Weight imitation loss by $\exp(A(s, a) / \beta)$ where $A$ is the estimated advantage.
- **Metric**: Faster convergence to smooth driving trajectories.

### [ ] C92. Residual RL Control on Top of Geometric Controller
- **Hypothesis**: Neural networks are great at perception but struggle with high-frequency control. Let a classical controller handle nominal driving, and let RL predict a residual correction $\Delta u$.
- **Method**: Control output $u = u_{pure\_pursuit} + \pi_{residual}(s)$.
- **Metric**: Trajectory tracking precision; passenger comfort index.

---

## Category 16: Comprehensive Benchmarking & Edge-Case Stress Testing (C93–C97)

### [ ] C93. Full 100-Route Bench2Drive Official Validation
- **Hypothesis**: A 20-route subset has sampling variance. Full 100-route evaluation across all 5 towns provides definitive scientific benchmarking.
- **Method**: Execute full 100-route evaluation protocol with official Bench2Drive metrics.
- **Metric**: Official Leaderboard Driving Score (DS), Route Completion (RC), and Infraction Penalty.

### [ ] C94. Town05 Multi-Lane Highway & High-Speed Merge Stress Test
- **Hypothesis**: Urban towns (Town01–03) lack high-speed on-ramps and multi-lane highway weaving. Town05 stresses high-speed navigation.
- **Method**: Evaluate on 20 dedicated Town05 highway routes at 90 km/h.
- **Metric**: Lane-change collision rate; merging success rate without deceleration.

### [ ] C95. 50-Vehicle Dense Gridlock & Urban Traffic Stress Test
- **Hypothesis**: Low-density evaluation hides intersection gridlock vulnerabilities.
- **Method**: Spawn 50 autonomous background vehicles and 30 pedestrians in Town03 dense downtown.
- **Metric**: Zero deadlock occurrences; mean velocity maintained above 15 km/h.

### [ ] C96. Camera Frame Drops & Communication Lag Simulation
- **Hypothesis**: Real-world camera hardware drops frames and incurs communication latency over CAN bus.
- **Method**: Inject random 1–2 camera frame drops and 100ms artificial latency during closed-loop evaluation.
- **Metric**: Trajectory deviation and recovery stability under lag.

### [ ] C97. Human Driver Benchmark Comparison
- **Hypothesis**: Establishing human performance on identical Bench2Drive routes provides an absolute upper-bound reference for the thesis.
- **Method**: Record 5 expert human teleoperation trials on the 20-route subset and compute comparative statistical metrics.
- **Metric**: Statistical parity with human Driving Score, smoothness, and route completion.

---

# PART 2: PHASE 2 — Multimodal Hardware Sensors & Beyond-Vision Expansion
*(Deferred: Physical LiDAR, 4D Radar, Hardware Depth, HD-Maps & Multimodal Fusion)*

> [!NOTE]
> All experiments in this phase require physical hardware sensors beyond the 3-camera + waypoints budget. These experiments are deferred until Phase 1 vision-centric milestones are achieved.

---

## Category 17: Multimodal Hardware Sensors & Beyond-Vision Expansion (P2-1–P2-15)

### [ ] P2-1. Range-View LiDAR Cylinder Projection Fusion
- **Hypothesis**: Raw point clouds are computationally heavy, but projecting 32-beam LiDAR into a 32x512 range-view image allows standard 2D convolution and early feature fusion with RGB.
- **Method**: Construct a 2-channel range-view image (depth, intensity) and fuse with RGB features via a shared residual stage.
- **Metric**: Obstacle detection recall at 30m+; distance estimation error reduction.

### [ ] P2-2. Sparse 4D Radar Cross-Attention (mmWave Doppler)
- **Hypothesis**: Cameras are easily degraded by heavy rain and fog, whereas radar point clouds provide Doppler radial velocities invariant to weather.
- **Method**: Encode sparse radar returns (azimuth, range, radial velocity) with a lightweight PointNet and cross-attend into ego-state tokens.
- **Metric**: Closed-loop driving stability under HardRain and Foggy presets.

### [ ] P2-3. Raw 3D LiDAR Point-Cloud VoxelNet / PointNet Fusion
- **Hypothesis**: Cylinder projection loses fine 3D vertical geometry. Direct 3D voxelization (e.g. VoxelNet / PointPillars) preserves precise object bounding boxes.
- **Method**: Voxelize 32-beam point clouds into $0.1m \times 0.1m \times 0.15m$ pillars and fuse with visual tokens via cross-attention.
- **Metric**: Pedestrian 3D localization accuracy at 20m+.

### [ ] P2-4. Hardware Active Depth Camera Fusion (ToF / Stereo IR)
- **Hypothesis**: Monocular depth estimation struggles with untextured walls and extreme lighting. Active hardware depth sensors provide dense metric distance directly.
- **Method**: Feed calibrated depth camera frames as a 4th channel $(R, G, B, D)$ into the vision backbone.
- **Metric**: Lead-vehicle distance error < 0.1m across all lighting conditions.

### [ ] P2-5. Vectorized HD-Map Polyline Tokenization & Cross-Attention
- **Hypothesis**: Rasterized map overlays require convolutional decoding; vectorized lane polylines provide exact topological connections with minimal tokens.
- **Method**: Tokenize lane boundaries and centerlines from an external HD-map sensor as sequenced polyline coordinates with vector direction embeddings.
- **Metric**: Junction lane-following accuracy; zero wrong-lane infractions.

### [ ] P2-6. Ground-Truth Semantic Segmentation Mask Pre-Tokenizer
- **Hypothesis**: Raw RGB includes irrelevant background visual noise (sky, foliage textures). Direct semantic segmentation sensor streams force the model to focus purely on drivable road, vehicles, and pedestrians.
- **Method**: A/B test training on 13-class CARLA semantic segmentation masks vs raw RGB frames.
- **Metric**: Generalization gap across different visual town styles.

### [ ] P2-7. Multi-Modal Dynamic Gating & Sensor Dropout (RGB + LiDAR + Radar)
- **Hypothesis**: Multi-modal models often over-rely on a single modality (e.g. LiDAR) and fail catastrophically if that sensor drops frames.
- **Method**: Apply random modality dropout ($p=0.15$ for RGB, $p=0.15$ for LiDAR, $p=0.15$ for Radar) during training with dynamic gating.
- **Metric**: Policy robustness when any sensor modality is dropped during evaluation.

### [ ] P2-8. Multi-Modal Radar/LiDAR Redundant Autonomous Emergency Braking (AEB)
- **Hypothesis**: Single-modality vision AEB can be tricked by optical illusions (e.g. road billboard images of cars). Hardware radar/LiDAR verification provides fail-safe redundancy.
- **Method**: Require 2-out-of-3 sensor consensus (Camera, Radar, LiDAR) before suppressing AEB triggers.
- **Metric**: False-positive braking rate reduced to 0.00% while maintaining 100% collision prevention.

### [ ] P2-9. 360-Degree Surround Sensor Ring (6+ Cameras + 4 Radars + LiDAR)
- **Hypothesis**: 3 forward-facing cameras leave rear and side blind spots, limiting aggressive highway lane changes.
- **Method**: Equip vehicle with 6 surround cameras (Front, Front-Left, Front-Right, Back, Back-Left, Back-Right) and 4 corner radars.
- **Metric**: Zero side-swipe collisions during high-speed highway lane changes.

### [ ] P2-10. V2X Infrastructure Telemetry & Cooperative Perception
- **Hypothesis**: Occluded cross-traffic cannot be seen by on-vehicle sensors until entering the intersection. V2X transmits roadside unit (RSU) trajectories.
- **Method**: Ingest RSU actor bounding boxes as auxiliary sequence tokens in the decision transformer.
- **Metric**: Blind intersection collision rate reduced to absolute 0.0%.

### [ ] P2-11. Full Multimodal Teacher Distillation (TransFuser++ 360 LiDAR + Radar Rig)
- **Hypothesis**: Training a giant multimodal teacher with access to LiDAR point clouds and radar provides an optimal supervisory signal.
- **Method**: Train full TransFuser++ teacher on the complete sensor suite, then distill its feature representations into the vision-only student.
- **Metric**: Student Driving Score improvement on complex Town03/04 intersections.

### [ ] P2-12. Thermal / Long-Wave Infrared (LWIR) Camera Fusion for Night Navigation
- **Hypothesis**: RGB cameras fail in pitch-black rural roads without streetlights. LWIR thermal cameras highlight pedestrians and animals by body heat.
- **Method**: Fuse thermal infrared sensor stream with RGB tokens via cross-attention.
- **Metric**: Night-time pedestrian detection recall improvement from 45% to 99%.

### [ ] P2-13. Ultrasonic Proximity Sensor Fusion for Tight Urban Parking & Creeping
- **Hypothesis**: Cameras have close-proximity dead zones (<0.5m around bumpers). Ultrasonic sensors provide high-frequency near-field distance.
- **Method**: Ingest 8 ultrasonic distance readings into the controller fallback layer for tight maneuvering.
- **Metric**: Near-field scraping and bumper collision rate reduced to zero.

### [ ] P2-14. Asynchronous Multi-Modal Sensor Timestamp Synchronization & Drift Compensation
- **Hypothesis**: LiDAR spins at 10Hz, cameras shutter at 20Hz, and Radar reports at 50Hz. Unsynchronized timestamps cause spatial misregistration at high speeds.
- **Method**: Implement motion-compensated ego-pose interpolation to project all sensor returns to a common reference timestamp $t_0$.
- **Metric**: Multimodal spatial registration error < 0.02m at 90 km/h.

### [ ] P2-15. Event-Based Neuromorphic Sensor Fusion for High Dynamic Range Glare
- **Hypothesis**: Emerging from dark tunnels into bright sunlight causes severe camera sensor saturation (complete white-out). Event cameras provide 120dB dynamic range.
- **Method**: Fuse asynchronous event camera spikes with RGB frames using an event-to-tensor graph neural network.
- **Metric**: Zero lane departures during sudden tunnel exit illumination changes.
