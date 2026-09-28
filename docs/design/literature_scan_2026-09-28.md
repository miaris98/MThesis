# Literature scan, 2026-09-28: sample-efficient deep RL and camera-only driving

Why: find techniques and architectures, published up to Sept 2026, that bear on the open problems in
`docs/todo/TODO_ACTIVE.md`. The new TODO items are A14-A21 and B8-B14 there. Checked first against
`challenges/tried_and_ruled_out.md` and the older C-list, so nothing below repeats an experiment we
already ran. Numbers are the papers' own. Items marked *(unverified)* were not checked in the paper text.

The most important result came from our own code, prompted by LEAD's "intent asymmetry" (section 2.1).

---

## 1. Where the field is

### Atari 100k (26 games, 100k agent steps; mean human-normalised score unless noted)

| Method | Year | Type | Score |
|---|---|---|---|
| EfficientZero V2 | 2024 | MCTS + learned model (Gumbel search, mixed value target) | mean **2.428**, median 1.286 |
| EfficientZero | 2021 | MCTS + value prefix + SimSiam consistency | mean 1.943, median 1.09 |
| BBF | 2023 | model-free, scaled value net + resets | **IQM 1.045** (~6 GPU-h) |
| TWISTER | 2025 | transformer world model + action-conditioned CPC | mean **1.62** (best without look-ahead search) |
| DIAMOND | 2024 | diffusion world model in pixel space | mean 1.46 |
| STORM / DreamerV3 | 2023 | transformer / RSSM world model | mean 1.27 / 1.12 |
| IRIS | 2023 | discrete-token transformer world model | mean 1.046 |
| UniZero | 2024 | transformer latent world model + MCTS | on par with or above MuZero-style agents, single-frame input |
| OC-STORM | 2025 | STORM + object masks (SAM2/Cutie, 6 labelled frames per game) | beats STORM on 18/26 games |

Our position: EZ-V2 official, Breakout seed 0 = 321.2 at 100k (paper 400.1). At 10k our port scores
GTrXL 20.2 / ResNet 15.8 (6 seeds each) against official EZ-V2 10.1 (5 seeds). The 100k runs (B4) are in progress.

### Bench2Drive (220 routes, CARLA 0.9.15)

| Method | Sensors | DS |
|---|---|---|
| TransFuser v6 / LEAD (CVPR 2026) | cameras + LiDAR + radar | **95.0** (camera-only 360 deg variant **91.6 +/- 0.7**) |
| SimLingo / CarLLaVA (CVPR 2025) | 1 front camera, VLM | 85.94 |
| TF++ (PDM-Lite dataset paper, 2024) | camera + LiDAR | 84.21 |
| ORION (2025) | camera, VLM | 77.74 |
| Raw2Drive (NeurIPS 2025) | camera, model-based RL | best RL method *(DS unverified)* |
| TCP, ThinkTwice, DriveAdapter, DriveTransformer, UniAD, VAD, AD-MLP | various | in the official README table (an image, not extracted here) |

Our position: 19-route b2d20 subset, arm E e15 65.4 vs the original WoR 66.9 (tied, S-089/S-090).
The 219-route A9 run gives the first number comparable to this table. **Report it both ways:**
crashed routes scored 0 over 220 (the official convention: "Failed/Crashed status is also acceptable",
the JSON must hold 220 routes), and crashed routes excluded (our paired comparison).

---

## 2. Camera-only driving: findings mapped to our problems

### 2.1 Obstacle-in-lane routes (TODO A1): a train/test mismatch in our route input (S-096)
LEAD names three learner-expert asymmetries: visibility, uncertainty and **intent**. Intent means
"the student's intent is under-specified ... unaware of complex multi-lane maneuvers". Checking our
pipeline against that turned up a concrete defect:

- PDM-Lite's `autopilot.py` logs **two** routes: `route` (after `shift_route_around_actors` /
  `shift_route_for_invading_turn`, i.e. **already bent around** construction cones, parked cars,
  accidents and hazards) and `route_original` (the unmodified plan), plus a `changed_route` flag.
- `src/training/wor_dataset.py` reads `meas.get("route")`: **the shifted route**. It is the route-point
  input and the route overlay painted into the image.
- At evaluation `scripts/eval/bench2drive_agent.py` builds the route from the leaderboard's global
  plan (`set_global_plan`), which **goes straight through the obstacle** (= `route_original`).

So in training the route input tells the model where to swerve, and at test time it doesn't.
**Scope (S-097):** obstacle scenarios exist only in PDM-Lite's Town12/13 archives. Our 6-town set has none, so
for arms A-I `route` == `route_original` and they fail obstacle routes simply because they never saw one. The
mismatch explains why the one run that did see them (8-town, S-063) still drove into them. The fix needs both:
the Town12/13 obstacle archives (~70 GB) **and** `route_original` as input. **Fix (A14):** train on `route_original` as input/overlay,
and use the shifted `route` only as a supervision target (TF++'s "path checkpoints", A15).

### 2.2 Output representation (A15)
- TF++ (Hidden Biases, ICCV 2023): waypoints are ambiguous. It separates the **path** (space-indexed)
  from the **speed** (classified).
- CarLLaVA: path prediction cut layout collisions **0.68 -> 0.0**. SimLingo: disentangled path + speed
  "reduces static object collisions".
- We predict 5 time-indexed waypoints plus a two-hot target speed. The path head is the missing part.

### 2.3 Data selection (A16)
- CarLLaVA trains on "buckets" of interesting frames (acceleration, steering, hazards, red lights,
  walkers, **swerving obstacles**) plus one whole-dataset bucket: 2.9M -> 650k samples per epoch.
- The PDM-Lite dataset-bias paper keeps frames whose target speed changes by >0.1 m/s or path angle by
  >0.5 deg (~40%) plus 14% random: -49% data, same or better DS. Removing speed-class weights raised DS
  82 -> 85, and they use two-hot speed labels. Both of those are already how our `target_speed_loss` works.
- Unlike S-063's extra towns, this is about emphasis inside the data we have, not coverage.

### 2.4 Route and intent input, overlays (A17)
- LEAD: three target points (previous, current, next) as an explicit token instead of one: **+2.03 DS**.
  Removing the GRU waypoint-decoder bottleneck: +2.32 DS.
- Hecker et al. (ECCV 2018): a **rendered route-planner video** beats raw GPS coordinates as the route
  input. Rendered overlays work best as independent binary channels (no colour ambiguity).
- RAP (2025): rasterised 3D primitives instead of photoreal images, plus counterfactual recovery views,
  #1 on NAVSIM v1/v2, Waymo E2E and Bench2Drive. That argues semantic fidelity matters more than
  photorealism for planning inputs.
- Our overlay is painted onto the RGB image that a **frozen** RGB backbone reads. A separate route
  channel/token path, a map inset or target-point markers are cheap variants. Test them only after A14,
  since the overlay itself currently carries the mismatch.

### 2.5 Novel-architecture candidates for the CARLA side
- **LAW (ICLR 2025), latent world model as a self-supervised auxiliary task:** predict the next frame's
  latent features from the current latents + the predicted waypoints, supervised by the next frame's
  extracted latents. Perception-free setting, CARLA Town05 Long: DS **67.9 +/- 2.1 -> 70.1 +/- 2.6**,
  NAVSIM PDMS 77.5 -> 84.6. With our frozen backbone the target (next frame's regnety grid tokens) is
  fixed, so there is no collapse risk and the loss only shapes the policy head. It fits the WoR theme
  (a learned world model) (A18).
- **Multimodal trajectory heads:** DiffusionDrive (CVPR 2025, truncated diffusion from anchors, 2
  denoising steps), VADv2 / Hydra-MDP-style trajectory vocabularies. An obstacle in the lane is
  bimodal (stay / swerve), and a regression head averages the two modes. Combined with a latent world
  model scoring the candidates, this becomes an "imagine-and-select" planner (World4Drive / WoTE line) (A19).
- **Teacher alignment:** DriveAdapter (ICCV 2023) trains an adapter so student features match a frozen
  teacher planner's input space, masking teacher features where the teacher breaks rules. Town05 Long
  61.7 vs TCP 57.2. This is the concrete recipe for A7 (TF++ as teacher).
- **RL fine-tuning in closed loop:** CaRL (CoRL 2025): PPO with a single simple reward (route
  completion, infractions terminate or scale it multiplicatively) scales to 300M CARLA samples, where
  shaped rewards break at large batch. Residual RL around an imitation prior: CRAFT (2026), CLEAR
  (2026), OPTED (2026). Raw2Drive (NeurIPS 2025) is the only RL method on Bench2Drive (A20).

### 2.6 Collisions (A11)
- LEAD: PDM-Lite reacts to occluded actors and uses noiseless velocities of other cars, so its
  demonstrations are "successful but dangerous" or non-causal from the camera's view. Restricting the
  expert to camera-visible actors gave +1.37 DS. Without re-collecting data, frames where the expert
  reacts to an actor outside our single front camera's view can be down-weighted (A21).
- PDM-Lite dataset-bias paper: changing the expert's braking style for pedestrians (sharp, ~4 m, when
  clearly visible) cut collisions ~4x.
- CarLLaVA's main failure modes are rear-end collisions and high-speed merging, the same families as
  our 26401 / 27532 losses.

### 2.7 Things that did **not** help others (keep deprioritised)
- Temporal input: CarLLaVA 90.40 -> 90.37. Our C14/C19 stay unscheduled.
- A rear camera: CarLLaVA 90.40 -> 88.81. For surround cameras the evidence is mixed: LEAD's
  camera-only 360 deg variant is 91.6, but with a different expert and data (A13).
- Ego-status shortcut (Li et al., CVPR 2024): planners that get ego speed lean on it. Worth a
  speed-input check in the A1/A11 diagnosis (does the target speed stay at 0 once stopped?).

---

## 3. Sample-efficient deep RL: findings mapped to our Atari port

### 3.1 Why GTrXL beats ResNet (B5): tokenization, not necessarily attention
- "Don't flatten, tokenize!" (2025): the gain of SoftMoE in value-based deep RL comes from **tokenizing
  the encoder output** (one token per spatial position, "PerConv"), not from the experts. One scaled
  expert matches SoftMoE-4. Shuffling token order is worst, so spatial structure matters. Global average
  pooling also beats flattening, but less.
- "Mind the GAP" (2025): the encoder-to-dense connection is the scaling bottleneck, and GAP is a simple
  fix.
- Our GTrXL reads the 36 spatial tokens, while the EZ-V2 ResNet heads flatten. B5 must separate
  tokenization from attention: arms ResNet+flatten (now), ResNet+GAP, ResNet+tokens with a per-token
  MLP / single expert (no attention), and GTrXL.

### 3.2 More learning per sample (B8)
- SR-SPR (ICLR 2023) and BBF (ICML 2023): periodic **resets** let the replay ratio go to 8-16 without the
  primacy bias.
- BBF's recipe:
  - replay ratio 8 (2 as the cheap setting);
  - reset every 40k gradient steps: conv layers shrink-and-perturb **50%** of the way toward a random
    init, later layers fully reset;
  - n-step annealed 10 -> 3, discount 0.97 -> 0.997;
  - AdamW, weight decay 0.1;
  - a target network, SPR, and a 4x wider Impala CNN.
- BBF is a constant +0.45 IQM above SR-SPR at every replay ratio, and gains grow with the replay ratio.
- EZ-V2 runs about 1.2 updates per env step with no resets. That is the most documented lever not
  yet in our port.

### 3.3 Value and reward heads (B9)
- "Stop Regressing" (ICML 2024): **HL-Gauss** (Gaussian-smoothed categorical target) beats two-hot, MSE
  and C51 on Atari. EZ-V2 and our port use two-hot. It is a small code change, and our reward head has a
  miscalibration history (S-058, S-067).
- DreamerV3's related robustness tricks: symlog, two-hot, 1% unimix, percentile return normalisation.

### 3.4 Wall-clock (B10)
- Our 10k logs show `time c/t/u` ~6/71/23%: **reanalyze targets take ~71%**.
- ReZero (2024): backward-view reuse of child values plus periodic whole-buffer reanalyze cuts search
  time with equal or better scores.
- V-MCTS (Virtual Expansions, 2022): adaptive search budget.
- Seeds are our bottleneck for claims (B1 power analysis), so a 2x faster run means 2x more seeds.

### 3.5 Representation objectives and architectures (B11, B12, B14)
- TWISTER (ICLR 2025): action-conditioned CPC over future latents, with crop-resize augmentation,
  beats next-state prediction for transformer world models: 1.62 mean without search. Our port uses
  EZ-V2's SimSiam one-step-per-unroll consistency. A multi-step contrastive version for a search agent
  is untested in the literature (B11).
- UniZero (2024): a transformer over the latent **history** (KV cache) inside MuZero-style planning.
  Our GTrXL attends over space only (B12).
- OC-STORM (2025) and ObjectZero (2026, object-centric world model + MCTS with a GNN): object-level
  inputs from a pretrained segmenter plus a few labelled frames per game. An "overlay"-style input for
  Atari, and a novel-architecture direction (B14).

### 3.6 Plasticity at long budgets (B13, conditional)
- Lyle et al. (2024): LayerNorm + weight decay together keep plasticity in Atari.
- SimbaV2 (2025): hyperspherical normalisation + weight projection + a categorical critic.
- Only relevant if B4 shows late-training stagnation. S058e's LayerNorm failure was under the broken
  prioritized replay, so it does not rule this out under uniform replay.

---

## Sources
- Atari 100k overview: https://www.emergentmind.com/topics/atari-100k-benchmark
- BBF: https://arxiv.org/abs/2305.19452 ; SR-SPR: https://openreview.net/pdf?id=OpC-9aBBVJe ; Primacy bias: https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf
- EfficientZero V2: https://arxiv.org/abs/2403.00564 ; ReZero: https://arxiv.org/abs/2404.16364 ; V-MCTS: https://arxiv.org/abs/2210.12628
- UniZero: https://arxiv.org/abs/2406.10667 ; TWISTER: https://arxiv.org/abs/2503.04416 ; DIAMOND: https://arxiv.org/abs/2405.12399 ; IRIS: https://arxiv.org/abs/2209.00588
- Don't flatten, tokenize!: https://arxiv.org/abs/2410.01930 ; Mind the GAP: https://arxiv.org/abs/2505.17749 ; SoftMoE for RL: https://arxiv.org/abs/2402.08609
- Stop Regressing (HL-Gauss): https://arxiv.org/abs/2403.03950 ; DreamerV3: https://arxiv.org/abs/2301.04104
- Plasticity (LN + WD): https://arxiv.org/abs/2402.18762 ; SimbaV2: https://arxiv.org/abs/2502.15280
- OC-STORM: https://arxiv.org/abs/2501.16443 ; ObjectZero: https://arxiv.org/abs/2601.06604
- Hidden Biases of E2E Driving Models (TF++): https://arxiv.org/abs/2306.07957 ; Hidden Biases of E2E Driving Datasets (PDM-Lite): https://arxiv.org/abs/2412.09602
- CarLLaVA: https://arxiv.org/abs/2406.10165 ; SimLingo: https://github.com/RenzKa/simlingo
- LEAD / TransFuser v6: https://arxiv.org/abs/2512.20563
- LAW: https://arxiv.org/abs/2406.08481 ; DiffusionDrive: https://arxiv.org/abs/2411.15139 ; RAP: https://arxiv.org/abs/2510.04333
- DriveAdapter: https://arxiv.org/abs/2308.00398 ; CaRL: https://arxiv.org/abs/2504.17838 ; Raw2Drive: https://openreview.net/forum?id=CAz7UGRdLs
- CRAFT: https://arxiv.org/abs/2605.04470 ; CLEAR: https://arxiv.org/abs/2607.02841 ; OPTED: https://arxiv.org/abs/2609.20756
- Route rendering as input (Hecker et al.): https://arxiv.org/abs/1803.10158 ; Ego status (Li et al.): https://arxiv.org/abs/2312.03031
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- PDM-Lite code checked for `route` vs `route_original`: carla_garage `team_code/autopilot.py` (the `data = {...}` dict in the save step) and `privileged_route_planner.py` (`shift_route_*`)
