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

## 4. External report "Advancements in Sample-Efficient Deep RL (2022-2026)" (received 2026-09-28): what we take

The report (AI-generated overview, pasted by the user) covers plasticity loss, world models (IRIS, Delta-IRIS, TWISTER,
DIAMOND), EZ-V2/EZ-M, benchmarks beyond Atari 100k, and proposes a "G-DLAN" architecture. Several figures could
not be checked (e.g. "AltNet 52x over SAC", "CAPO 30x"), so none of them is cited without reading the paper.

**Taken:**
- *Measure plasticity before fixing it*: dormant-neuron ratio / FAU and effective rank -> **TODO B15** (offline, cheap).
- *Recurrent/gated nets resist plasticity loss at high replay ratio* -> **TODO B16**: GTrXL vs ResNet across replay
  ratios. This connects the thesis trunk choice to plasticity.
- *Plasticity interventions* (BBF soft resets, SR-SPR, ReDo, CReLU, FIRE, AltNet) -> variants of B8/B13, tried in that
  order of cost. AltNet (a twin network with hard resets) doubles memory and compute. Only consider it if B8's resets
  show post-reset score drops.

**Not taken, and why:**
- *DIAMOND / Delta-IRIS / pixel-space world models:* a different agent family (policy trained in imagination, no
  search). Replacing the EZ-V2 port would discard B4's curves, and the compute (days per game) is beyond our budget.
  TWISTER's useful part is already B11.
- *EZ-V2 continuous control, EZ-M multi-task, "gradient-guided MCTS":* Breakout's actions are discrete, and the
  thesis has no continuous-control task. Multi-task only with B6 (more games), much later.
- *G-DLAN* (Delta-Transformer + diffusion verifier + triple AltNet + gradient-guided continuous MCTS): four heavy
  systems, none of them tested in our setting, and aimed at continuous control. It cannot be built and ablated
  within a master's budget, and it does not address either thesis question (beat WoR; GTrXL vs ResNet trunk).
- *New benchmarks* (Crafter, Procgen, HumanoidBench, ViZDoom, MemoryMaze, SMAC, text games, SWE-bench) and the
  RL-for-LLMs section: out of scope. Generality within Atari is B6.
- The report has nothing on driving, so the CARLA plan (A9 -> A14/A16/A15) is unchanged.

---

## 5. Second pass, 2026-09-28 evening: items not covered above

Checked against sections 1-4, `TODO_ACTIVE.md` (A0-A21, B1-B16), the C-list and `tried_and_ruled_out.md`.
New TODO items: A22-A24, B17-B18. Additions to existing items: A6, A9, B3, B5, B8.

### 5.1 CARLA
- **Direct control head next to the waypoints (A22).** Hydra-NeXt (2025, front + back cameras, ResNet-50) on
  Bench2Drive: the trajectory-only planner (Hydra-MDP) scores **52.80 DS / 30.73 SR**, adding a control decoder
  (throttle/steer/brake classified over several future steps, focal loss on brake) gives ~65.5, and trajectory
  refinement gives **65.89 / 48.20**. Its abilities: overtaking 64.4%, emergency brake 61.7%, merging 40.0%. At
  inference, both outputs go through a kinematic bicycle model, and the closest proposals are averaged (brake by
  threshold). TCP (2022) was the first dual-branch design. "TCP-style dual-branch control" was on the ch. 13.27
  ranked list and never scheduled. Caveat: Hydra-MDP had no separate speed classifier, and ours does (two-hot
  target speed), so part of that gain may already be in our pipeline.
- **Scenario-supervised action experts (A23).** DriveMoE (CVPR 2026) on Bench2Drive: Drive-pi0 **55.85 DS**,
  Action MoE alone **67.31**, Vision MoE alone 68.68, both 74.22. It uses 1 shared + 6 experts, top-3 active,
  and trains the router with cross-entropy on the scenario/skill label, plus router noise against expert collapse.
  Its aim is to stop averaging across behaviour modes. Our PDM-Lite archives are named by scenario, so the labels
  are free.
- **Shadow-mode expert takeover (A6 recipe).** TakeAD (2025): PDM-Lite runs in shadow mode while the policy
  drives the Bench2Drive training routes. It takes over for 2 s when it predicts a collision or when
  |steer difference| > 0.2. It collected 8,596 / 6,106 / 5,799 / 6,395 / 6,078 samples over 5 rounds. Each round
  is DAgger for 1 epoch, then SimPO/DPO for 10 epochs (preferred = expert action, rejected = the policy's top-1,
  beta 0.1, gamma 0.1). VAD-style base: **61.82 -> 71.39 DS**, saturating after round 4. Cost: 18 h/round on 6 A800.
- **Offline post-training survey (2026, arXiv 2607.08072).** Four families: distillation (CRAFT, PaIR-Drive),
  preference alignment (DriveDPO, TakeAD), RL (world-model or rule rewards), and test-time reranking
  (DriveCritic). DriveDPO, CRAFT and TakeAD's preference stage need no simulator in the loop. The survey itself
  was read only as a summary.
- **The attention-vs-tokenization confound applies to CARLA too (A24).** From our code: arm H's `cnn` head
  (`SpatialQHead`, `src/models/world_on_rails/wor_policy.py`) is a 3x3 conv stack followed by
  `AdaptiveAvgPool2d(1)`, and our transformer head attends over the 4x4 regnety grid. So arm H (-15.0 DS vs E,
  S-090) shows "transformer head > WoR head", but it mixes global attention with keeping spatial tokens. This is
  the same confound "Don't flatten, tokenize!" found for SoftMoE on Atari (B5).
- **Bench2Drive ability scores (A9 addition).** `tools/ability_benchmark.py -r merge.json` (after
  `tools/merge_route_json.py`) reports Merging / Overtaking / Emergency Brake / Give Way / Traffic Sign, and
  `tools/efficiency_smoothness_benchmark.py` reports efficiency and comfort. It needs exactly 220 routes, with
  Failed/Crashed allowed. Every recent paper reports these (Hydra-NeXt, DriveMoE), and none of our scripts compute
  them.

### 5.2 Atari
- **B3, from the code (2026-09-28):** our `EZReplay` differs from EZ-V2's `ez/data/replay_buffer.py` in two
  places that matter under alpha = 1:
  1. EZ-V2 samples a batch **without replacement** (`np.random.choice(..., p=probs, replace=False)`), and ours
     samples **with replacement** (`np.random.choice(p.size, size=B, p=p)`). A few high-priority reward windows
     can then fill one batch many times over. That fits S-067's "reward-rich batches" and a reward head fitted
     to them.
  2. New data enters at EZ-V2's **current** buffer maximum (`self.priorities.max()`, which falls as priorities
     are updated) or at self-play priorities. Ours uses an **all-time** running `max_prio` that never falls. After
     an early value spike (S058a/b: V ~10x the MC return), every new transition enters with that stale, very
     large priority, and sampling piles onto the newest data.

  In both codebases the IS weights scale the whole loss. SpeedyZero (ICLR 2023, built on EfficientZero) adds
  **Priority Refresh**, a periodic recompute of stale priorities that is reported to stabilise value training,
  with lower variance than uniform sampling or DPER. Only the project page was readable, so the details are
  *(unverified)*.
- **Path consistency (B17).** GW-PCZero (NeurIPS 2023, built on the EfficientZero code): the value estimates
  (accumulated reward + discounted value) along the search's best path should be equal. The loss enforces this,
  down-weighting uncertain nodes. It reports **198% mean HNS vs EfficientZero's 194% at 25% of the compute**,
  worse than EZ on 2 of 26 games. Untested with Gumbel search or EZ-V2's mixed value target. The exact loss form
  and weights were not read (the PDF did not parse) *(unverified)*.
- **Attention entropy and seed variance (B18).** Zhai et al. (ICML 2023): transformer training instability
  coincides with attention-entropy collapse. sigma-Reparam (spectral-normalised linear layers plus a learned
  scale), or QK-normalisation, prevents it and makes training robust across seeds. Our GTrXL's 10k seed SD is
  11.3 vs ResNet's 3.5, and that spread is what blocks the trunk claim (B1: ~57 seeds per trunk at the current SD).
- **Hare & Tortoise (B8/B13 variant).** Lee et al. (ICML 2024): a slow EMA copy (tortoise) of the trained
  network (hare), with the hare periodically reset to the tortoise. It keeps plasticity without BBF's post-reset
  score drop, and the paper reports gains on Atari-100k agents. EZ-V2 already keeps a target network (hard copy
  every 200 updates) that could serve as the tortoise.
- **Status note:** B4 seed 0 in our harness: GTrXL **299.0 at 40k** (EZ-V2's own run 278.2) and ResNet
  **43.0 at 30k** (EZ-V2 92.0; 61.4 at 20k). The eval episodes still repeat (3-4 distinct scores per 10), so
  these are single-seed, B2-caveated numbers.

### 5.3 Not taken
- *Fail2Drive* (200 paired routes, 17 shift types, -22.8% average success under shift) and *Bench2Drive-Robust*:
  generalisation and robustness claims come after "beats WoR", and each costs a full evaluation campaign.
- *Open-loop proxies for closed-loop DS* (NAVSIM/Bench2Drive correlation study, 2026): Spearman 0.90 over only 8
  methods, with Ego Progress the best single predictor. It ranks methods, not checkpoints of one method, and our
  own result (open-loop loss can't pick checkpoints, ch. 13.28) stands.
- *VLAAD / CARLA-Collide* (TF++ 84.21 -> 86.97 DS with a VLM collision-risk token): needs a VLM input. The usable
  part, training on our own closed-loop failure clips, is TakeAD's recipe (A6).
- *DriveDPO*: needs a trajectory vocabulary and rule-based scores per anchor, so it comes after A19.
- *MAD-TD* (model-generated data to stabilise high update ratios): continuous control. EZ-V2's reanalyze
  already uses the model.
- *SpeedyZero's Clipped LARS*: for large distributed batches. We run batch 256 on one GPU.

---

## 6. Third pass, 2026-09-28 night: visual overlays, physics priors, data curation

Prompted by the question "can visual overlays, or ideas from physics and maths, make learning more
sample-efficient?". New TODO items: A25-A27, B19. Additions to A17 and B14.

### 6.1 Why overlays are a strong lever in our CARLA pipeline
- With a **random** backbone the policy still scored 50-60 DS (S-072). The route overlay painted on the image (plus
  the speed scalar) carries much of the driving signal. Spatial information survives even random conv features.
- Our overlay (`src/config/route_overlay.py`) is a single green polyline, alpha 0.55, 5 px, with optional lane-width
  guide lines. It encodes direction only: no distance along the route, and no ego state.

### 6.2 Evidence that drawn cues beat the same information as numbers
- **AimBot (CoRL 2025)** draws the robot's own state into the image: a "shooting line" to the projected stopping
  point of the end-effector, a reticle whose size grows as it nears the target, and colour for the gripper state.
  It needs <1 ms and no architecture change. Results:
  - LIBERO-Long: pi0 85.2 -> 91.0, pi0-FAST 81.6 -> 87.1, OpenVLA-OFT 87.5 -> 91.2.
  - Real world (50 trials each): pi0 27 -> 43, OpenVLA-OFT 21 -> 36, pi0-FAST 42 -> 47.
  - The ablations show the policy reads the cue: **proprioception alone 85.2 vs overlay 91.0**, and randomised cues
    drop it to 77.4. Plain colour (84.0) and a fixed-size reticle (84.6) are worse than the default (87.1), so
    colour coding and a distance-dependent size both carry information.
- **RT-Trajectory (ICLR 2024)** and **HAMSTER (2025)**: policies conditioned on a 2D path drawn into the image
  generalise better than language- or goal-conditioned ones. HAMSTER draws the path with a **colour gradient
  for temporal progress** and compares overlay with concatenated channels.
- **Atari counter-evidence, "Virtual Augmented Reality" (2023):** SAM segmentation added to PPO's input helped
  4/12 games (Beam Rider, Seaquest, Chopper Command, Space Invaders, +101-129%) and **not Breakout or Pong**, at
  ~500x the training time. Generic segmentation overlays are not a Breakout lever (B14 note).

### 6.3 Physics priors
- **Our expert brakes by a physics model, IDM.** carla_garage `team_code/config.py`: for a lead vehicle,
  minimum gap `s0 = 4.0 m`, time headway `T = 0.25 s`, max acceleration 24.0, comfortable braking 8.7 / 3.72 m/s²
  (low / high speed, threshold 6.02 m/s), exponent 4. IDM's desired gap is `s* = s0 + v*T + v*dv / (2*sqrt(a*b))`
  (`dv` = closing speed), and braking dominates once `(s*/s)^2` is large. The static part `s0 + v*T` depends only
  on ego speed, so it can be **drawn**:
  - 6.5 m at 10 m/s, 9.0 m at 20 m/s (from `s0 + v*T`);
  - the gap where IDM's braking term reaches the comfortable deceleration, about `s* * sqrt(a/b)` ~ 2.5 s*
    (~16 m at 10 m/s; our own arithmetic from the config values).

  Drawing these marks turns the expert's longitudinal rule into a visual comparison between the lead car and a
  mark (A25).
- **Time-to-contact (tau) theory (Lee, 1976):** drivers brake so that tau-dot, the rate of change of time to
  contact from the lead car's optical expansion, stays near **-0.5**. Measured means are -0.51, with critical
  values -0.44 to -0.52. Large-scale driver data (NOVA, 2026) finds the looming signal **1/TTC** in both braking
  and lane-change decisions. 1/TTC is bounded at 0 when nothing closes in, so it is a better-conditioned target
  than raw distance (C28). One frame cannot measure expansion rate, which is the physical case for giving the
  model two frames on the lead-vehicle question specifically (A26).
- **Symmetry (Noether-style priors):** Pong and Breakout are near left-right symmetric (Geometric Coherence,
  2026). Equivariant MuZero (DeepMind, 2023) proves that equivariant networks make MuZero's entire action
  selection equivariant, tested on ProcGen maze rotations, not Atari. EqR (ICML 2022) and SiT (2024) apply
  equivariance/invariance to Atari-100k. A reflected Breakout transition with LEFT/RIGHT swapped is another valid
  transition: free data under a fixed budget (B19).

### 6.4 Data curation (sample efficiency from better data)
- **CUPID (CoRL 2025):** influence functions estimate each demonstration's effect on **closed-loop return**. It
  filters out harmful demos and picks the most useful new ones. With <33% of the data, curated this way, it
  reaches state-of-the-art diffusion policies on RoboMimic, with similar gains on hardware. Related: TracIn / TRAK
  gradient-similarity attribution. As far as this scan found, it has not been applied to end-to-end driving (A27).
- **Keyframe weighting (Wen et al., ICML 2021):** up-weighting expert action changepoints fixes the "copycat" failure
  of history-conditioned behaviour cloning, shown in CARLA. It is needed if A26 adds a second frame.

### 6.5 Not taken
- *Koopman Dreamer (2026):* spectrally bounded rotation-scaling latent dynamics. It cuts 64-step latent MSE by 89%
  vs DreamerV3 and wins 8/9 DMC proprioceptive tasks. EZ unrolls 5 steps and searches 2-5 deep, and Breakout's
  bounces are discontinuous, so the long-horizon drift it fixes is not our failure mode.
- *Second-order optimisers (Shampoo/SOAP):* no RL evidence found. We also inherit EZ-V2's SGD recipe.
- *FLARE (latent flow, NeurIPS 2021):* explicit latent differences beat pixel frame stacking on DMC (1.9x RAD at
  500k). EZ-V2 already stacks 4 frames, and the evidence is model-free continuous control. Revisit only for A26's
  two-frame question.
- *Game-specific physics overlays for Atari* (e.g. drawing the ball's predicted landing point): this hand-codes the
  solution and breaks comparability with EZ-V2's Atari-100k numbers.

---

## 7. Fourth pass: mathematics and physics in more depth

New TODO items: A28-A30. Additions to B2, B13, B19.

### 7.1 Decision theory on the speed head (A28), from our code
- `TargetSpeedHead.expected_speed` (`src/models/world_on_rails/aux_heads.py`) drives the PID with the
  softmax-weighted **mean** of the bins (0, 4, 8, 10, 13.9, 16, 17.8, 20 m/s). The class's own docstring gives the
  reason for classifying: a multi-modal target's mean is "the one speed the expert never drives". Inference
  brings that averaging back. P(stop) = P(8 m/s) = 0.5 gives 4 m/s, a creep toward whatever caused the stop.
- Bayes decision theory: under absolute loss the optimal point decision is the **median**. Under an asymmetric
  linear (pinball) loss it is the **tau-quantile**, with tau = c_slow / (c_slow + c_fast). A quantile below 0.5
  encodes "too fast costs more than too slow". The median snaps to a mode whenever one holds >50% of the mass.
- Temporal consistency without temporal input: an **HMM forward filter** over the speed bins,
  `p_t ∝ softmax_t ⊙ (A^T p_{t-1})`, with a sticky transition matrix A, smooths single-frame flicker at inference
  only. It adds no history to the network, so no copycat shortcut. Temporal ensembling of *actions* fails on
  multi-modal outputs (section 5); filtering a categorical posterior is the version that respects modes.
- Prior warning: the hard "brake on uncertainty" rule deadlocked 3 routes (S-059). A median is not a hard
  threshold, but low quantiles move toward one.

### 7.2 Physics as a grey box: the expert's own IDM on predicted state (A29)
- PDM-Lite's longitudinal rule is IDM with published constants (section 6.3). A **grey-box** controller predicts
  the physical state the rule needs (lead-vehicle gap `s`, closing speed `dv`) and applies IDM analytically. It
  then extrapolates the way the physics does, where a network would only interpolate between training frames.
- **Physics-informed car-following** (PIDL-CF, Transportation Research C 2021): IDM encoded as a computational
  graph next to a neural net is more accurate than either alone and is reported to be more data-efficient when
  data are sparse.
- **BarrierNet / differentiable control barrier functions** (T-RO 2023; vision-based driving, 2022): a
  differentiable QP safety layer on network-predicted state (lane offset, obstacle position). Obstacle
  avoidance: crash rate **53% -> 28%** with predicted state, and **3%** with ground-truth state. The safety gain
  is bounded by how well the state is perceived. That is why A29 needs the C28 / A26 heads to be accurate first.
- An IDM gap constraint `h = s - s0 - v*T >= 0` is a control barrier function. The CBF condition
  `dh/dt >= -alpha*h` gives a closed-form speed cap, which is the same family as IDM's braking term.

### 7.3 Statistics: what "beats WoR on Bench2Drive" needs (A30)
- S-090's power analysis used the SD of per-route differences (21 DS for E). That treats the routes as a
  random sample from a population of routes (**super-population** inference: "does E drive better in general?").
- For the benchmark claim, the 220 routes are **fixed**. The estimand is the mean over those routes, and its only
  randomness is **run-to-run noise within each route** (design-based / finite-population inference, Neyman). The
  CI width then depends on `sigma_run`, not on route heterogeneity.
- ~~Rough size: sigma_run ~ 4-8 DS, giving +/-1.2-2 DS at 220 routes and +/-4-6 at 19.~~ **Measured in section 8.1
  (S-100): too optimistic.** Our arms' run noise is 10-20 DS per route. About 80% of the per-route E-vs-WoR variance
  is run noise, so the fixed-set framing narrows the 19-route CI only a little (E: +/-7.9 vs +/-9). What helps is
  repeating the few noisy routes (Neyman).
- **Neyman allocation:** spend repeat runs where run noise is large, with repeats per route proportional to that
  route's `sigma_run`. Coin-flip routes get repeats, while routes that always score 100 (or always ~23) need one.
- Report both estimands. The fixed-set one answers the benchmark question; the super-population one answers
  generalisation.

### 7.4 Atari
- **Flat minima (SAM), B13 variant.** PLASTIC (NeurIPS 2023), Atari-100k IQM: DrQ 0.258; +SAM 0.325; +reset 0.343;
  +LayerNorm 0.259; +CReLU 0.256; all four 0.421. SAM costs ~2x per update, and LN + reset alone ("PLASTIC-dagger")
  reports 0.536. SAM is the one "smoothness of the loss surface" lever we have not considered.
- **Mirror equivariance (B19 step 2), two code facts.** The GTrXL position embedding is a learned absolute table
  (`ez_model.py`, `self.pos`), which breaks mirror symmetry. The dynamics network encodes the action as one scalar
  plane `a/|A|` (MuZero's convention), which makes the discrete actions ordinal. Swapping LEFT/RIGHT is then not
  a linear map on the input, so the equivariant version needs one-hot action planes and a mirror-tied position
  table.
- **Variance components (B2 addition).** The eval SE counts repeated episodes as independent (S-068). A hierarchical
  seed x episode model separates between-seed from within-seed variance. It says whether the next GPU-hour should
  buy seeds or episodes (Neyman allocation again). At GTrXL's seed SD, seeds almost certainly win.

### 7.5 Theory that supports what we already do
- **Compounding error with continuous actions** (Simchowitz et al., COLT 2025; follow-up arXiv 2507.09061): smooth
  deterministic imitators can suffer error exponential in the horizon. Two regimes:
  - **Action chunking** helps only when the dynamics are open-loop stable, and hurts without a low-level
    stabiliser.
  - **Noise-injected demonstrations** help when the dynamics are closed-loop stable: 50% clean + 50% noisy
    trajectories, with large noise (0.4-0.5 of the normalised action range), matching DAgger without queries.
    Supervision is needed only along the "excitable" directions.

  Our recovery camera (random lateral + yaw offset per route) is the observation-space version of this, along a
  car's two excitable directions (lateral offset, heading). That is theoretical support for keeping it (S-073),
  not a new item. Our PID on replanned waypoints is the low-level stabiliser the chunking result asks for.

### 7.6 Not taken
- *MICo / bisimulation (Kantorovich) metrics:* evaluated on DQN-family and SAC agents over 60 games. EZ's SimSiam
  consistency already shapes the latent, and B11 covers the next representation objective.
- *Noether networks (meta-learned conserved quantities):* shown for video prediction of physical systems. Breakout's
  "conserved" ball speed changes with hits, and the evidence for RL is missing.
- *Physically-based fog from logged depth (Koschmieder's law):* cheap and principled, but nothing points at weather
  (C4, C50). Reopen only if A3 shows weather-related failures.

---

## 8. Fifth pass: the maths done on our own data, plus camera geometry (S-100)

Scripts are in the session scratchpad (`a30.py`, `slope.py`) and not yet in the repo. Inputs: the valid 19-route
eval JSONs on `E:\MThesis_EXP\live_2026092{5,6,7}_*` (b19_* random-backbone runs and the brake/creep run excluded;
"couldn't be set up" = harness failure, dropped), and `bench2drive220.xml`.

### 8.1 Run-to-run noise (A30, measured)
Pooled within-route SD across repeated runs of the same checkpoint:

| Agent | sd_run (DS) | dof |
|---|---|---|
| WoR (runs on boxes V, X2, and a partial on S) | **8.6** | 27 |
| D e15 (3 runs) | 10.3 | 27 |
| A SWA (2 runs) | 10.5 | 12 |
| A e15 (repeats) | 18.7 | 24 |
| B e15 (repeats) | 20.4 | 23 |
| G e20 (2 runs, 6 routes) | 23.8 | 6 |

- **Our arms are ~1.2-2.4x noisier per run than WoR.** On the 7 routes where A e15, A SWA, D e15 and WoR all have
  repeats, the SDs are 23.4 / 13.4 / 15.5 / 12.8. The SWA-vs-A e15 variance ratio (3.1) is **not** significant
  (route-swap permutation p = 0.25), so "weight averaging makes the closed loop less noisy" is a hint, not a
  finding.
- **Run noise is concentrated on a few routes.** Ours, pooled: 23687 **41.5**, 2143 **32.0**, 3717 19.0 (WoR 29.8),
  3936 14.8, 2286 14.1, 2050 11.8, 26401/27532 ~9.5. It is ~0 on 3373, 24340, 24784, 24795, 25318, 25975 and 28198.
- **Decomposition:** E's per-route difference SD vs WoR is 21 (S-090). Run noise alone predicts
  `sqrt(16.7^2 + 8.6^2) = 18.8`, so ~80% of that variance is run noise and the true per-route effect varies by only
  ~9 DS.
- **E e15 - WoR = -1.0 on 19 routes:**
  - route-population CI [-9.8, +8.1];
  - fixed-set, analytic with pooled variances: [-8.9, +6.9];
  - fixed-set, run-bootstrap [-6.2, +2.2]. This one is anti-conservative: two runs per route underestimate the
    variance, and zero-variance routes contribute nothing.
- **At 220 routes, one run each:** fixed-set +/-2.5 DS, and +/-1.9 with two runs of our agent. The
  route-population version gives +/-2.8. Both are roughly the "+/-3 at ~190 routes" of S-090. The route set is not
  the lever. **Run noise is**, and repeats on the noisy routes (Neyman allocation) buy the most precision per CPU
  hour.
- Other arms under the fixed-set estimand: A e15 -9.5 [-15.7, -3.3] and B e15 -9.0 [-15.4, -2.5] are *worse than
  WoR* with CIs excluding 0. A SWA -5.2 [-11.9, +1.5] and D e15 -1.9 [-7.4, +3.6] are not.

### 8.2 Where the DS goes: we collide, WoR stalls (A28)
Lost DS per route on the 19 routes, split by the metric's own formula (`100 - DS = (100 - RC) + RC*(1 - P)`):

| Agent | DS lost | not finishing | penalties | main infractions |
|---|---|---|---|---|
| E e15 | 34.6 | 16.7 | **18.0** | 7 vehicle collisions, 2 stop, 2 layout, 1 red light |
| D e15 | 35.4 | 16.5 | **18.9** | 8 vehicle collisions, 2 red light, 2 stop, 2 layout |
| WoR | 32.2 | **27.8** | 4.4 | 3 layout, 1 yield-to-emergency, 0 vehicle collisions |

Bench2Drive's `statistics_manager.py` leaves `MIN_SPEED_INFRACTION` "unused": no penalty, despite ~300 events per
agent. The two agents sit at **opposite ends of a speed-risk trade-off**: WoR is cautious and times out, we finish
and collide. The metric prices a vehicle collision at x0.60 of the route's score, while slowness costs nothing
until it reaches the 200 s route cap. So the Bayes cost ratio in A28 is far from symmetric, and the knob A28 adds
(median / low quantile) moves us along exactly this trade-off. Converting half our penalty loss (~9 DS) into
caution without losing completion would put E well above WoR on these routes. This is the quantitative case for
A28 before any new training.

### 8.3 Does the flat-ground overlay drift on hills? No (checked, not an item)
The overlay projects route points onto the ego's ground plane (`project_ego_points`, z = 0). The car pitches with
the road under it, so the error is the road's height deviation from that tangent plane, projected:
`dv = f*dh/(x + 1.5)`, with `f = 358.5 px` at 1024 wide. From the 2 m-spaced route waypoints of all 220 routes:

| Town group (routes) | median / p95 / max grade | error 30 m ahead: median / p95 / max |
|---|---|---|
| training towns (44) | 0.0 / 2.5 / 23% | 0.0 / 4.7 / 23.3 px |
| Town12/13 (151) | 1.5 / 3.0 / 5.1% | 1.1 / 5.1 / 12.9 px |
| other unseen (25) | 0.0 / 3.5 / 7.0% | 1.1 / 5.7 / 13.2 px |

The p95 error is about one line-width (5 px at full resolution, ~2.5 px at our 512-wide input), and the unseen hilly
towns are no worse than the training towns. A slope-aware overlay is **not** needed. The z rounding (0.1 m) puts the
measurement noise at the same few pixels.

### 8.4 Perspective: how big is the car we must brake for? (A31)
Input pipeline: 1024x512 render, 1024x384 crop, resized to 512x192, so `f = 179 px` at input resolution. A car
(1.85 m wide) at distance D from the camera spans `331/D` px, which is **`10.4/D` cells of the stride-32 map** the
head reads. IDM's braking-onset gap (~2.5 x (4 m + 0.25 s * v), A25) plus ~3.9 m from the camera to the front bumper
puts **D ~ 19-27 m at 30-72 km/h**. There the lead car covers **0.4-0.55 of one feature cell**, and it is then
averaged with ~5 other cells into one of the 16 tokens (each token spans 4x1.5 cells). At the stride-16 and
stride-8 levels of the same frozen backbone, which are already computed in the same forward pass
(`CarlaPretrainedEncoder.forward_pyramid`), it spans ~0.8-1.1 and ~1.5-2.2 cells. Arm E (288x768, 1.5x) raised this
to ~0.6-0.8 cells and is our best arm (65.4), a weak hint that resolution at the lead car matters. FOVEA (ICCV 2021)
magnifies where objects are expected and raised streaming AP on Argoverse-HD from 17.8 to 23.0.

### 8.5 Atari: symmetric relative position bias (B20)
GTrXL's learned absolute position table breaks two symmetries that a convolution trunk has for free: **translation**
(the ball bounces the same way anywhere on the screen) and, with B19, **mirror**. An attention bias that depends only
on the offset `(|dx|, dy)` between tokens keeps both. On the 6x6 grid that is 6 x 11 = 66 scalars per head, in
place of today's 36 learned position vectors, which can encode any absolute layout. ConViT (ICML 2021) initialises attention to be conv-like (local) with a learned gate to escape locality, and
reports much better sample efficiency than DeiT in low-data regimes. That matches our 100k-sample regime and the
GTrXL design, which already starts as the ResNet (zero-initialised output) and must earn its contribution.

---

## 9. Sixth pass: the physics of our failure modes, from our own eval data (S-101)

Scripts: `bifurc.py`, `collide.py` (session scratchpad), on the same valid 19-route JSONs as section 8.

### 9.1 The run noise is a collision lottery, and it has a closed form
On every noisy route, route completion is ~100 in nearly all runs. What differs between runs is the **number of
vehicle collisions k**, and each one multiplies the route's score by 0.60. If collisions on a route arrive as a
Poisson process with rate `lam`, the probability generating function gives:
- `E[DS] = RC * E[0.6^K] = RC * exp(-0.4 * lam)`;
- `Var[DS] = RC^2 * (exp(-0.64 * lam) - exp(-0.8 * lam))`. This peaks at `lam = ln(1.25)/0.16 ~ 1.4`, where the
  run-to-run SD is **~29 DS** from the collision count alone.

This is why our arms are 1.2-2.4x noisier than WoR (section 8.1): WoR has almost no vehicle collisions (k ~ 0, no
lottery). **Cutting the collision rate therefore raises the mean *and* narrows every CI** (A30).

Two regimes, from the per-route dispersion of k across our 6 arms' runs:
- **Systematic, underdispersed** (var k << mean k): the same collision happens in nearly every run. This is a
  policy error, not luck, and training can fix it:
  - 26401: mean k 1.00, var 0.22;
  - 27532: 0.91 / 0.09;
  - 25318: 1.00 / 0.00;
  - 2664: 1.11 / 0.11;
  - 2050: 0.71 / 0.22.
- **Lottery, Poisson or overdispersed:** these routes create the run noise:
  - 23687: 1.75 / 1.84;
  - 3457: 0.62 / 0.84;
  - 2143: 0.64 / 0.55;
  - 3717: 0.85 / 0.64.

**Ceiling:** removing vehicle collisions at unchanged route completion would add **+15.0 DS per route** to our
arms' runs on average. That is optimistic, because more caution costs completion, but it is 10x the E-vs-WoR gap.

### 9.2 Two physical signatures of repeated collisions
16 runs had >= 2 vehicle collisions. In 12 of 21 consecutive pairs it is the same actor:
- **~5 m apart, the same parked car** on ParkedObstacleTwoWays (2664, 3457): the car pushes into the stationary
  obstacle again. This is the creep that averaging "stop" with "go" produces (A28), not a perception failure.
- **~33-60 m apart, the same moving car** (mostly ~39 m) on **23687 HighwayExit**, in 5 of 6 arms: repeated contact
  with one vehicle while crossing lanes to the exit. This is consistent with a car alongside, outside the front
  camera's +/-55 deg view, which the policy keeps steering into. WoR (4 cameras) scores 100 on both runs, and E
  e15 scored 100 once. First concrete evidence for A13 (side cameras) and A21 (expert reacts to actors we cannot
  see). It is not proven: the eval JSON has the collision position but not the ego's pose.

### 9.3 Atari: the training objective is not the evaluation metric (B21)
- Our port trains on **sign-clipped** rewards (`clip_reward=True`, EZ-V2's recipe) and is scored on **raw** points.
  Breakout's rows are worth 1 / 4 / 7 from bottom to top, so a clipped agent counts bricks while the score weights
  top bricks 7x. The top (orange/red) rows also speed the ball up, which costs lives. The clipped objective sees
  that risk but not the extra reward.
- The value-prefix and value heads already use MuZero's invertible h-transform with supports over h-space
  [-300, 300]. That transform was introduced (Pohlen et al. 2018) precisely to **drop** reward clipping, and MuZero
  trains on raw Atari rewards. EfficientZero / EZ-V2 / LightZero keep both, a contradiction raised in LightZero
  issue #239 and never answered there, with no ablation reported.
- Evidence it matters for Breakout is thin: DQN-family agents find the tunnel under clipping anyway. The test is
  cheap (one flag), but only meaningful at 30k+ steps, when top rows are reached.

### 9.4 Atari evaluation: the benchmark protocol has no sticky actions (B2 correction)
The Atari-100k protocol used by SPR, EfficientZero and BBF evaluates **without** sticky actions. B2's plan (30+
episodes with sticky p = 0.25) is right for a like-for-like comparison inside one harness and as a robustness
check. Comparisons with EZ-V2's published 400.1 need the no-sticky protocol, with more episodes and distinct no-op
starts.
Breakout scores are also **bimodal** once the tunnel regime appears (GTrXL s0 at 30k: episodes of 62-92 vs 288-310).
Like CARLA's collision lottery, the mean then mixes two regimes. Report the tunnel rate (P(score >= 200)) and the
median next to the mean.

---

## 10. Seventh pass: energy, latent space, Gaussian noise, input transformations, vectorisation

New TODO items: A32, B22, B23. Additions to A28 and B18.

### 10.1 Energy
- **Our categorical heads are already energy models.** `softmax(logits)` is a Boltzmann distribution with energy
  `-logit` and temperature 1. Two consequences:
  1. **Temperature is a free, fittable parameter.** Temperature scaling (Guo et al. 2017) fits one scalar `T` on
     held-out NLL. A28's median/quantile decoding is only meaningful on a calibrated distribution, so fit `T` first.
  2. **Energy-based decoding with a physics cost.** Bayes decision: pick bin `j` minimising `sum_i p_i C(i, j)`.
     A physical cost of being too fast is the **extra stopping distance** from surplus kinetic energy,
     `(v_j^2 - v_i^2) / (2a)` with a = 4.95 m/s² (PDM-Lite's `brake_acceleration`). The cost of being too slow is
     lost progress, `(v_i - v_j) * T`.

  **Result of working the example (a caution):** with P(0) = P(8 m/s) = 0.5 and T = 1 s, the expected cost of
  4 m/s is `0.81w + 2` (w = weight on a metre of stopping-distance deficit). The decision depends on w:
  - for `0.83 < w < 2.5`, 4 m/s beats both stopping (cost 4) and 8 m/s (cost `3.23w`);
  - below 0.83 it picks 8 m/s;
  - above 2.5 it stops.

  Over a wide middle range the convex (v²) kinetic-energy cost therefore *recommends creeping*. Creeping is physically cheap unless the gap is already ~0, which is exactly our 5 m repeat hits on
  parked cars (S-101). So the stop/go decision needs either a committing rule (median / mode / tau-quantile, A28) or
  a gap-aware cost (A29's predicted gap). Kinetic energy alone is not enough.
- **Attention is an energy minimiser.** A modern Hopfield network's update is attention (Ramsauer et al., 2020),
  and the inverse temperature beta decides whether it retrieves one pattern (low attention entropy) or averages
  many. B18's attention-entropy collapse is beta growing too large, and QK-normalisation fixes beta.
- **Energy-based (implicit) policies** (Implicit BC, Florence et al. 2021) represent multi-modal actions such as
  swerve vs stay. Later work (Diffusion Policy, Chi et al. 2023) found them harder to train than diffusion or
  classification heads. A19's trajectory vocabulary is the discrete, stable version of the same idea. No new item.

### 10.2 Latent space
- **Atari: the latent is unbounded.** EZ-V2 turns MuZero's min-max state normalisation off for Atari
  (`state_norm: False` in `atari_breakout.yaml`). Our port copies this, so only BatchNorm bounds the latent that
  the dynamics network feeds back into itself. Our failures (S-058) happened in imagined states (phantom reward at
  depth 1, value inflation), and GTrXL's seeds diverge (B18).
- **SimNorm (TD-MPC2, ICLR 2024)** projects the latent onto a product of simplices: softmax over groups of 8
  channels at a fixed temperature. The latent is then bounded and sparse by construction, and the paper's
  ablations call SimNorm and LayerNorm **essential for training stability**. "Simplicial embeddings improve sample
  efficiency" (Lavoie et al.) is the representation-learning origin. The evidence is continuous control, not Atari,
  so it is a screen, not a claim (B22).
- CARLA: the head reads frozen, cached backbone features, so the backbone's latent geometry is fixed. The head-side
  transformation that matters is how the *low-dimensional* inputs enter (10.4).

### 10.3 Gaussian noise
- **Gaussian label smoothing on the CARLA speed head.** HL-Gauss (B9, "Stop Regressing") is the cross-task version:
  a Gaussian-smoothed categorical target instead of two-hot, integrated over TF++'s non-uniform bins. It gives the
  better-calibrated distribution that A28 decodes. Cheap, and folded into A28.
- **Noise-injected demonstrations** (DART; the 2025 theory in section 7.5) remain the principled use of Gaussian
  noise for imitation, but the PDM-Lite data is already collected. Online, this is TakeAD's shadow mode (A6).
- **Not taken: isotropic Gaussian noise on cached CARLA features.** The perturbations that help us are photometric
  (colour, arm D) and geometric (recovery camera). Neither is isotropic in feature space, and nothing found shows
  isotropic latent noise standing in for them. It would only be a way to keep the feature cache, and the cheaper
  fix there is caching a few colour-augmented copies.
- **Not taken: Gaussian noise on EZ latents during unrolls.** No evidence for search-based agents. SimNorm addresses
  the same drift with evidence.

### 10.4 Input transformations: how scalars and coordinates enter the CARLA head (A32)
- In `qwen_wor_policy.py`, **speed enters as `Linear(1, d)` on the raw scalar** (`speed_proj`). The speed token is
  `s*w + b`, a straight line in embedding space: every speed-dependent behaviour has to come from that one direction
  after normalisation. The **route enters as `Linear(2N, d)` on raw metre coordinates** (`route_proj`).
- This is the textbook setting for **spectral bias** (Rahaman 2019; Tancik et al. 2020): networks fed raw
  low-dimensional coordinates learn high-frequency detail slowly. Examples here are a 1-2 m lateral route shift
  at 30 m, or the speed threshold where braking starts. Diffusion models embed their scalar timestep sinusoidally
  for this reason.
- **Direct evidence for imitation learning:** "Fourier Features Let Agents Learn High Precision Policies with
  Imitation Learning" (June 2026) encodes metric 3D coordinates with **fixed, log-spaced** sinusoids (L = 16,
  wavelengths 4 m to 2 cm). RoboCasa success rose 13.2% -> 33.9%, and the real world 14.8% -> 40.2% over 44 tasks.
  Log-spaced frequencies beat random Gaussian ones, and the result is robust to L and the minimum wavelength.
- Polar route coordinates (log distance, bearing) match the pure-pursuit geometry the controller uses (steering
  from the bearing to a lookahead point) and could share the same encoding. Try it second.

### 10.5 Vectorisation
- **The search is launch-bound by design.** In `gumbel_mcts.py`, the selection loop runs `bool(active.any())`
  (a GPU-to-host sync) inside a loop that is itself inside the 16-simulation loop, and each step issues many small
  kernels. Reanalyze is 71% of our wall-clock (B10), and it runs this search on every sampled state.
  - **Sync-free:** loop to the known bound (depth <= simulation index + 1) with the existing `active` mask instead
    of breaking early.
  - **CUDA graphs:** with no data-dependent control flow, each simulation step is fixed-shape and can be captured
    with CUDA graphs (`torch.cuda.graphs` / `torch.compile(mode="reduce-overhead")`).
  - DeepMind's `mctx` runs Gumbel MuZero search as fixed-shape, fully jitted loops for this reason.
- **Vectorise over seeds.** The binding constraint for every Atari claim is the number of seeds (B1: ~57 per trunk
  at GTrXL's spread). PureJaxRL trains 2048 PPO agents in half the time of one PyTorch agent by `vmap`-ing whole
  runs. Our EZ networks are small, so K seeds in one process help: stacked parameters via `torch.func`
  (`stack_module_state` + `vmap`, or grouped convolutions), one batched search over K x B roots, CPU envs for all
  seeds in one vector env. That turns one GPU into K runs, instead of the two we fit today by running separate
  processes (B23).

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
- Second pass (section 5): Hydra-NeXt: https://arxiv.org/abs/2503.12030 ; DriveMoE: https://arxiv.org/abs/2505.16278 ; TakeAD: https://arxiv.org/abs/2512.17370 ; post-training survey: https://arxiv.org/abs/2607.08072 ; DriveDPO: https://arxiv.org/abs/2509.17940
- GW-PCZero: https://github.com/CMACH508/GW_PCZero (paper: NeurIPS 2023, https://openreview.net/forum?id=vHRLS8HhK1) ; SpeedyZero: https://sites.google.com/view/speedyzero ; Hare & Tortoise: https://arxiv.org/abs/2406.02596 ; attention entropy collapse / sigma-Reparam: https://arxiv.org/abs/2303.06296
- Fail2Drive: https://arxiv.org/abs/2604.08535 ; open-loop vs closed-loop correlation: https://arxiv.org/abs/2605.00066 ; VLAAD / CARLA-Collide: https://arxiv.org/abs/2603.25946 ; MAD-TD: https://arxiv.org/abs/2410.08896
- Third pass (section 6): AimBot: https://arxiv.org/abs/2508.08113 ; RT-Trajectory: https://arxiv.org/abs/2311.01977 ; HAMSTER: https://arxiv.org/abs/2502.05485 ; Virtual Augmented Reality for Atari: https://arxiv.org/abs/2310.08683 ; CUPID: https://arxiv.org/abs/2506.19121 ; Keyframe-focused IL: https://arxiv.org/abs/2106.06452 ; Fighting copycat agents: https://arxiv.org/abs/2010.14876
- Physics/maths: Lee 1976 tau theory: https://journals.sagepub.com/doi/10.1068/p050437 ; tau-dot braking test: https://pubmed.ncbi.nlm.nih.gov/7595250/ ; NOVA (1/TTC in car-following and lane change): https://arxiv.org/abs/2606.10583 ; PDM-Lite IDM parameters: https://github.com/autonomousvision/carla_garage/blob/leaderboard_2/team_code/config.py ; Equivariant MuZero: https://arxiv.org/abs/2302.04798 ; EqR: https://proceedings.mlr.press/v162/mondal22a/mondal22a.pdf ; SiT: https://arxiv.org/abs/2406.15025 ; Geometric coherence (Pong/Breakout symmetry): https://arxiv.org/abs/2602.02978 ; Koopman Dreamer: https://arxiv.org/abs/2607.19719 ; FLARE: https://arxiv.org/abs/2101.01857
- Fourth pass (section 7): BarrierNet: https://arxiv.org/abs/2111.11277 ; dCBF for vision-based driving: https://arxiv.org/abs/2203.02401 ; physics-informed car-following (PIDL-CF): https://arxiv.org/abs/2012.13376 ; Pitfalls of IL with continuous actions: https://arxiv.org/abs/2503.09722 ; action chunking + exploratory data: https://arxiv.org/abs/2507.09061 ; PLASTIC: https://arxiv.org/abs/2306.10711 ; MICo: https://arxiv.org/abs/2106.08229
- Fifth pass (section 8): FOVEA: https://arxiv.org/abs/2108.12102 ; ConViT: https://arxiv.org/abs/2103.10697 ; Bench2Drive penalty table: `Carla-utils/carla_garage/Bench2Drive/leaderboard/leaderboard/utils/statistics_manager.py` (`PENALTY_VALUE_DICT`, `PENALTY_PERC_DICT`) ; route elevation: `Carla-utils/carla_garage/Bench2Drive/leaderboard/data/bench2drive220.xml`
- Sixth pass (section 9): h-transform / transformed Bellman operator (Pohlen et al. 2018): https://arxiv.org/abs/1805.11593 ; LightZero clipping-vs-h-transform issue: https://github.com/opendilab/LightZero/issues/239 ; BBF (Atari-100k protocol, sticky actions): https://arxiv.org/abs/2305.19452 ; Bench2Drive collision criterion and penalties: `Bench2Drive/leaderboard/leaderboard/utils/statistics_manager.py`
- Seventh pass (section 10): Fourier features for IL: https://arxiv.org/abs/2606.12334 ; Fourier features (Tancik et al.): https://arxiv.org/abs/2006.10739 ; spectral bias of value approximation: https://openreview.net/forum?id=vIC-xLFuM6 ; TD-MPC2 (SimNorm): https://arxiv.org/abs/2310.16828 ; Simplicial embeddings: https://openreview.net/forum?id=mCpq1GCKxA ; Hopfield networks is all you need: https://arxiv.org/abs/2008.02217 ; Implicit BC: https://arxiv.org/abs/2109.00137 ; Diffusion Policy: https://arxiv.org/abs/2303.04137 ; temperature scaling (Guo et al.): https://arxiv.org/abs/1706.04599 ; PureJaxRL: https://github.com/luchris429/purejaxrl ; mctx: https://github.com/google-deepmind/mctx ; DART: https://arxiv.org/abs/1703.09327 ; Latent Policy Barrier: https://arxiv.org/abs/2508.05941
- EZ-V2 replay buffer checked for B3: `E:\MThesis_EXP\reference_code\EfficientZeroV2\ez\data\replay_buffer.py` (`save_trajectory`, `_prepare_batch_context`) against `atari_qwen/training/train_ez_offpolicy.py` (`EZReplay`)
