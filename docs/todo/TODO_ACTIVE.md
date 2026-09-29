# Active TODO (created 2026-09-25, rewritten 2026-09-27)

The short list of what is worth doing next. Older items are ticked in their own files with the reason
under each; fully closed TODO files are in `docs/archive/`. Before adding anything here, check
`challenges/tried_and_ruled_out.md`: if it is there, it has been done.

Status: **Running** / **Next** / **Later** / **Done**. Times are Athens. Item numbers are cited from
the challenges log, so they are never reused or renumbered; new items get the next free number.
The status block below is the only snapshot in this file: replace it, don't append new ones
(history goes to `challenges/`).

---

## Status: 2026-09-29 15:20 Athens (S-106, S-107) - **both boxes are destroyed at 22:00**

| Box | Job | ETA (Athens) |
|---|---|---|
| P1 (RTX 5090 32 GB, EPYC 9734) | B4 seed 0: GTrXL (`S058h_gtrxl_100k_s0_uniform`, eval 50k **320.0**) + ResNet (`S058i_resnet_100k_s0_uniform`, 50k **230.8**), resumed at 52k | 100k ~21:00 |
| P1 | A9 leftovers, 1 lane at Low quality (`lanes_0929e_P1.txt`, 25 runs) | ~17:30 |
| P2 (RTX 3090 24 GB, Xeon Gold 6222) | A14 arm J fine-tune `carla_armJ_ft_obst` (E e15 -> e20, `--route_key route_original`, 427k frames incl. obstacle archives, 18 min/epoch) | e20 ~16:15 |
| P2 | A9 leftovers, 1 lane at Low (`lanes_0929e.txt` line 1, 25 runs, 0.22x real time) | ~21:00 |
| P2 after arm J | arm J e20 eval, 3 lanes (`lanes_armJ.txt`, auto-start): b2d20's 19 routes first, then A9's 45 other obstacle routes | R19 ~18:30, obstacle ~21:30 |
| 21:30-21:55 | final pulls to E: + HF stage 2 (Atari finals, arm J checkpoints); resume state of anything unfinished to HF | before 22:00 |

A9 = full Bench2Drive (219 routes) for arm E e15 and the original WoR. P1 lost ~1.5 h to a crash loop (S-107): on the
RTX 5090 a `-quality-level=Low` map load segfaults at random (Epic never did), and the evaluator runs a lane's routes in
XML order, so crash routes blocked whole lanes. Every result so far is Low, so the leftovers stay at Low, mostly on
P2's 3090. Seed 1 (GTrXL ~12k, ResNet ~14k) is paused with its state on E: (`live_20260928_final_atari_resume\P1`).
Seed-0 resume state (52k, with replay) is also on the private HF repo `mthesis-relay` (`s0_20260929/`, sha-checked).
**Literature scan (2026-09-28):** new items A14-A21 and B8-B14, from `docs/design/literature_scan_2026-09-28.md`.
Second pass (evening, section 5 there): A22-A24, B17-B18, plus additions to A6, A9, B3, B5, B8. B3 now has two
concrete code differences from EZ-V2's replay buffer to test first. Third pass (overlays, physics, data curation,
section 6): A25-A27, B19, plus notes on A17 and B14. Fourth pass (maths/physics, section 7): A28 (speed head as
a Bayes decision), A29 (grey-box IDM), A30 (fixed-route-set statistics), plus notes on B2, B13, B19. Fifth pass
(the maths run on our own data, S-100, section 8): A30 measured (run noise, not the route set, is the lever; our
arms are 1.2-2.4x noisier than WoR); we lose 18 DS/route to penalties where WoR loses 28 to timeouts (A28's case);
the flat-ground overlay is fine on hills (no item); new A31 (foveal tokens) and B20 (symmetric position bias).
Sixth pass (S-101, section 9): run noise is a Poisson collision lottery with a closed form; systematic vs lottery
collisions; repeat-hit physics (creep into parked cars; side-swipes on HighwayExit, evidence for A13); new B21
(raw-score training through the existing h-transform); B2 protocol correction (Atari-100k has no sticky actions).
Seventh pass (energy, latent space, noise, transforms, vectorisation; section 10): A32 (sinusoidal speed/route
embeddings), B22 (SimNorm latent), B23 (sync-free CUDA-graph search, then several seeds per process); A28 gets
temperature scaling + HL-Gauss, plus a derivation showing a kinetic-energy cost alone favours creeping.
Eval-only additions (2026-09-28 night): A33 (ensemble the heads over the shared frozen backbone) and B24 (test-time
search scaling on saved checkpoints).
**Top CARLA lead (S-096, S-097):** our 6-town training set has **no** obstacle scenarios (they are all in Town12/13), so
arms A-I never saw one. The one run that had them (8-town, S-063) fed PDM-Lite's obstacle-shifted `route` as input,
while evaluation feeds the unshifted plan. The fix needs both: obstacle data **and** `route_original` (A14), then
moderate oversampling (A16).
Backups (2026-09-29): `E:\MThesis_EXP\live_20260929_boxP1_5090_atari_carla` and `live_20260929_boxP2_3090_carla`,
every 5 min: logs, `*.txt` lane plans, A9 results, guardian logs, Atari checkpoints (not replay buffers), MLflow, and
on P2 `checkpoints/carla_armJ_ft_obst`. Earlier sessions: `live_20260928_*` (Y, Z, AA, AB, final_a9, final_atari_resume).

**Order of work from here:**
1. *No box needed:* A9 merge (paired E vs WoR, official 220 with crashes = 0, seen/unseen towns, excluded routes)
   -> decides whether E ties WoR at scale; add Bench2Drive's ability scores (A9). A30 step 1 is done (S-100): plan
   A9's leftover session with repeats on the noisy routes (Neyman). A28's offline mode check. B15 plasticity diagnostic on the
   saved GTrXL/ResNet checkpoints, with B18's attention entropy on the same pass. HF stage 2.
2. *Next box session, CARLA (the thesis claim):* finish A9's leftover routes (1 box, a few hours), then A14 arm J
   (Town12/13 obstacle archives + `route_original`; cheapest first: fine-tune E e15 ~5 epochs), evaluated on b2d20 +
   A9's 50 obstacle routes, with a matched WoR-head arm; then A16 (arm K) and A15.
3. *Next box session, Atari:* resume GTrXL/ResNet s0 to 100k from the E: state (Ampere, bf16); seeds 1-2 fresh on
   Ampere. In spare headroom: B3's two replay switches (10k screen, 2 seeds). Then B9 (HL-Gauss, cheap), then B16
   (trunk x replay ratio, with B15's metrics logged).
   CARLA after arm J: A24 (attention mask, thesis claim), A25 (IDM braking-gap overlay, cheapest) and A22
   (control head) are one arm each. A26 step 1 (tau-dot before each collision) is eval-only and goes with A11.
   A28 (median / quantile speed decoding) needs no training: run it on the first eval box, on E e15.
Hardware: CARLA is CPU-bound (~5 cores/lane): pick a high CPU quota. Atari needs native bf16 (Ampere or newer).
Avoid Turing (2080 Ti: no bf16, and Vulkan hung on driver 595, S-099).
Result table of 2026-09-27 (all boxes destroyed then, S-092):

**Goal check (S-089, S-090):** original WoR **67.8 / 66.0** (2 runs, mean 66.9) on the 19 b2d20 routes.
Best arm **E e15 65.4**, -1.5 vs WoR, 95% CI [-10.6, +8.0]: **tied, goal not met.** TF++ 80.3.

| Result | Routes | DS | Note |
|---|---|---|---|
| WoR original run 1 / run 2 | 19 / 19 | 67.8 / 66.0 | 4 cameras vs our 1 (A13) |
| Arm H e15 (WoR cnn head, our pipeline) | 19 | **50.4** | -16.5 vs WoR, -15.0 vs E e15, both significant |
| Arm H e20 | 6 (partial) | 38.9 | WoR 50.4 on the same 6 - not needed, H is settled |
| Arm I e15 (288x768 + colour aug) | 12 (partial) | 58.5 | WoR 59.6, E e15 64.7 on the same 12 |
| Arm I e20 | 6 (partial) | 49.2 | WoR 50.4, E e15 55.7 on the same 6 |
| EZ-V2 official, 10k, seeds 0-4 (30 ep) | - | 7.0 / 15.1 / 26.9 / 1.3 / 0.0, mean **10.1** | our port: GTrXL 20.2, ResNet 15.8 (6 seeds) |

---

## CARLA

Where the points are (S-090): **5 obstacle-in-lane routes** that WoR and every arm fail (2509, 24795,
2664, 3457, 25318: 15-37 DS); **2 finished routes capped at 60 by one vehicle collision** in every arm
while WoR scores 100 (26401, 27532); junction routes we win (3936, 2050, 2143). 19 routes only
resolve differences of about 10 DS, so more small-effect arms cannot decide "beats WoR".

### A1. Why the car drives into the obstacle - **Next** (highest value, eval-only first)
Five routes, not three: 2509, 24795 (ConstructionObstacle), 2664, 3457 (ParkedObstacleTwoWays),
25318 (ParkedObstacle). All time out in every arm, did not improve with 8-town data that contains
these scenarios (S-063), and **WoR fails them too** (22-37), so solving them (~+15 DS) is the one
lever that would separate us from WoR on 19 routes. PDM-Lite's expert *leaves the lane* to pass; our
policy is conditioned on route points and a route overlay that go straight through the obstacle.
Check, from telemetry already recorded (`tel_armA_e15`, S-066) plus one new run with
waypoint/overlay dumps on these five routes:
- Do the predicted waypoints bend around the obstacle at all, or follow the route line?
- Does the target-speed head drop to 0 before the obstacle, or keep a creep speed into it?
- In the training frames of these scenarios, where are the route points relative to the expert's
  actual path? If the route runs through the obstacle while the expert swerves, the policy is being
  taught two conflicting signals.
Outcome decides the fix: route conditioning (e.g. route points from the expert's driven path in
lane-change segments), speed head, or controller (A4). Absorbs C5, C69, C78.
**Found 2026-09-28 (S-096, S-097):** (1) the 6-town set (Town01-05, 10) contains no obstacle-type scenario;
they are all in Town12/13 (~35 GB of obstacle archives per town). Arms A-I fail these routes because they never
saw one. (2) The 8-town run had them but trained on the shifted `route` input, while evaluation feeds the unshifted
plan (= `route_original`). When the data is on a box, check how often `changed_route` fires on obstacle
frames and how far `route` and `route_original` diverge. The fix is A14. Also check the
ego-speed shortcut (Li et al. CVPR 2024): does the target speed stay at 0 once the car has stopped?

### A14. Obstacle data + the unshifted route (`route_original`) - **Works: arm J e18 +15.1 vs E on obstacle routes [+5.0, +25.3]; next the matched WoR-head arm** (S-096, S-097, S-107, S-109)
2026-09-29: `carla_armJ_ft_obst` = E e15 fine-tuned to e20 (E's LR schedule, `--route_key route_original`, TF++
`all_towns/model_0030_0.pth` backbone, 427k train frames incl. the 20 obstacle archives). e16: ADE 0.284 m, lat 0.056 m.
Eval of e20 on P2 (`lanes_armJ.txt`, labels `j20_b2d20_*` / `j20_obst_*`). The matched WoR-head arm is still to do.
PDM-Lite logs `route` (bent around obstacles by `shift_route_around_actors` / `shift_route_for_invading_turn`)
and `route_original` (the plan the leaderboard gives at test time), plus a `changed_route` flag.
`wor_dataset.py` reads `route` for the route points **and** the route overlay, so on obstacle frames the training
input already shows the swerve, and at evaluation it doesn't. Our 6-town set has no obstacle frames at all
(S-097), so **arm J = arm E's config + the 20 Town12/13 obstacle-type archives (~70 GB, not all 234 GB of the
two towns) + `route_original` as the route input and overlay** (fall back to `route` where the key is missing).
Cheaper variant: arm E's input was already effectively `route_original` (no shifts in its towns), so fine-tune
E e15 for ~5 epochs on the mixed data instead of a 20-epoch retrain. Because the WoR baseline never saw these
scenario types, add a WoR-head arm on the same data for the matched comparison (retraining allowed, tuning not). Evaluate on the 19 routes and the
5 b2d20 obstacle routes (2509, 24795, 2664, 3457, 25318) and **A9's 50 obstacle-type routes** (29 in Town12/13,
21 elsewhere; report the two subsets separately) against E e15 and the original WoR from A9. It predicts a large effect on those 5
routes (~+15 DS if solved, S-090) and little change elsewhere. Pair it with A15 so the swerve is still
learned from a target rather than from the input.

### A15. Path output next to the speed output - **Next** (with A14)
TF++, CarLLaVA and SimLingo all predict a space-indexed **path** separately from speed. CarLLaVA: layout
collisions 0.68 -> 0.0; SimLingo: fewer static-object collisions. Target = PDM-Lite's shifted `route` (TF++'s
"path checkpoints", e.g. 10 points at 1 m spacing), lateral PID on the path, longitudinal control from the
existing two-hot target speed. Absorbs the "TF++ path checkpoints" line in `tried_and_ruled_out.md` and
part of A7.

### A16. Oversample obstacle/swerve frames and drop redundant ones - **Next** (arm K = arm J + sampler)
CarLLaVA trains on buckets (acceleration, steering, hazards, red lights, walkers, **swerving obstacles**, plus
a whole-dataset bucket; 2.9M -> 650k samples/epoch). The PDM-Lite dataset-bias paper keeps frames whose
target changes (>0.1 m/s or >0.5 deg) plus 14% random: -49% data, DS equal or better. Sampler weights from
`changed_route`, steering and target-speed changes. Different from S-063 (more towns = coverage): this is
emphasis. Our target-speed loss is already two-hot without class weights, as that paper recommends.
**Not possible on the 6-town set**: it has no obstacle frames (S-097), so this needs arm J's data. Oversample
**moderately**: the swerve frames (`changed_route`, plus a window of approach frames before it) up to ~10% of each
batch, inside a mixed sampler. Training mostly or only on obstacle frames risks false overtakes (swerving around
queued traffic or cars parked outside the lane, head-on collisions in the TwoWays variants where the expert
waits for a gap), forgetting other skills, and overfitting to 20 archives. Check that non-obstacle routes don't
regress (b2d20 minus the 5 obstacle routes).

### A22. Direct control head next to the waypoints (TCP / Hydra-NeXt) - **Next** (one arm, after arm J)
Today the PID turns 5 waypoints plus a target speed into throttle, steer and brake. Hydra-NeXt (camera-only,
Bench2Drive) adds a control decoder: throttle, steer and brake classified over a few future steps, with focal loss
on brake. The trajectory-only planner scores 52.80 DS; with the control decoder it reaches ~65.5, and with
refinement 65.89 (overtaking 64.4%, emergency brake 61.7%). TCP was the first dual-branch design. It was on the
ch. 13.27 ranked list and never scheduled.
Why here: short-horizon control reacts faster to other cars than a PID on 5 waypoints. That is A11's failure
family: rear-end collisions and merges on 26401/27532, the only routes where WoR beats us outright.
Plan:
- Add a head on the transformer state that classifies the expert's next-k controls. PDM-Lite's measurement files
  log its controls; check the key names on the data box, since `wor_dataset.py` reads none of them today.
- Fuse at inference: TCP's rule (control branch in turns, waypoints when straight) first, Hydra-NeXt's
  bicycle-model matching second.

Caveat: Hydra-MDP had no separate speed classifier and ours does, so part of the reported gain may already be in
our pipeline. Run A11's diagnosis first: if the collisions happen after the target-speed head already predicted a
stop, the PID or its lag is the problem, which argues for this item.

### A25. Draw the expert's braking gap into the image (IDM overlay) - **Next** (one arm; cheap)
PDM-Lite brakes for a lead vehicle by IDM (`carla_garage/team_code/config.py`: minimum gap s0 = 4.0 m, time
headway T = 0.25 s, max acceleration 24.0, comfortable braking 3.72-8.7 m/s²). Its desired gap is
`s* = s0 + v*T + v*dv/(2*sqrt(a*b))`, and the part `s0 + v*T` depends only on ego speed.
- **Overlay:** draw it on the road with the route-overlay code, as a short crossbar at s0 + v*T and a second one
  where IDM's braking term reaches comfortable deceleration (~2.5 x that, ~16 m at 10 m/s). Colour them by speed.
- **Why it could work:** the policy gets speed only as a scalar (`speed_mlp`), and the frozen backbone sees no
  speed-dependent geometry. With the bars drawn in, "is the lead car closer than my braking gap?" becomes a local
  visual comparison. That is the question behind A11's rear-end and merge collisions.
- **Evidence:** AimBot (CoRL 2025) draws the robot's own state into the image and beats the same state as
  numbers (85.2 -> 91.0 on LIBERO-Long; real world +16/50). Randomised cues fall to 77.4, so the policy reads
  them. Our route overlay already carries enough signal to drive at 50-60 DS on a random backbone (S-072).
- **Arm:** arm E + bars. Also run a control with the bars drawn at a random speed, AimBot's check that the
  policy reads them. Watch A11's routes (26401, 27532) and the collision count, not only DS.

Unlike S-096's route, speed and camera geometry are the same in training and evaluation, so there is no
train/test mismatch to create. Keep the bars thin and low-alpha so they don't hide the lead car. (Section 6 of
the literature note.)

### A31. Foveal tokens from a finer feature level where the route goes - **Later** (one arm; after A25)
Perspective arithmetic (S-100, literature note 8.4): at input resolution `f = 179 px`, so a car covers `10.4/D` cells
of the stride-32 map. At IDM's braking-onset distance (D ~ 19-27 m at 30-72 km/h) the lead car is **0.4-0.55 of one
feature cell**, then averaged with ~5 other cells into one of 16 tokens. The frozen backbone already computes
stride-16 and stride-8 maps in the same pass (`forward_pyramid`), where the same car spans ~1 and ~2 cells.
- **Arm:** project the route at ~15-35 m ahead with the overlay geometry. Crop a small window of stride-16 features
  around it, pool it into 4-8 extra tokens and give them their own type embedding.
- **Cost:** no extra backbone pass. A windowed stride-16 cache adds ~18k floats per frame.
- **Evidence:** arm E (1.5x resolution everywhere) is our best arm, a weak hint. FOVEA (ICCV 2021) magnifies
  expected object regions and raised streaming AP on Argoverse-HD 17.8 -> 23.0.
- **Targets:** A11 (reading the lead car's gap) and A1 (seeing the obstacle early enough to swerve). It is also
  what the C28/A29 lead-vehicle heads would read.

### A32. Sinusoidal embeddings for speed and route points - **Next** (one arm; small code change)
`qwen_wor_policy.py` embeds speed as `Linear(1, d)` on the raw scalar, and the route as `Linear(2N, d)` on raw metre
coordinates. The speed token is `s*w + b`, a straight line in embedding space. This is the spectral-bias setting
(Tancik et al. 2020): networks fed raw low-dimensional coordinates learn fine detail slowly. Examples here are a
1-2 m lateral route shift at 30 m, or the speed where braking starts.
- **Evidence:** "Fourier Features Let Agents Learn High Precision Policies with Imitation Learning" (June 2026) uses
  fixed, log-spaced sinusoids on metric coordinates (L = 16, wavelengths 4 m -> 2 cm). RoboCasa 13.2% -> 33.9%, real
  world 14.8% -> 40.2% (44 tasks). Log-spaced beats random Gaussian frequencies, and it is robust to L.
- **Arm:** arm E + `[sin, cos](2*pi*x/lambda_k)` for each route coordinate (lambda from ~64 m down to ~0.25 m) and for
  speed (~40 down to ~0.5 m/s), then the existing linear projections.
- **Variant:** polar route coordinates (log distance, bearing), matching the pure-pursuit geometry.
- **Watch:** obstacle routes after A14 (fine lateral shifts) and A11's rear-end collisions (speed thresholds).

The overlay already carries the route spatially, so the gain may be smaller than in point-cloud manipulation.

### A26. Looming (1/time-to-contact) as the lead-vehicle target - **Later** (with A5/C28)
Lee's tau theory (1976): drivers brake so that tau-dot, the rate of change of time to contact from the lead car's
optical expansion, stays near -0.5 (measured -0.51). Large driver datasets find the looming signal 1/TTC in
both braking and lane changes (NOVA, 2026).
1. **Diagnostic, eval-only (feeds A11):** before each rear-end collision, compute the lead car's TTC and tau-dot
   from telemetry and actor positions. Did we brake too late (tau-dot below -0.5 early) or too weakly?
2. **Target:** predict 1/TTC of the lead vehicle from PDM-Lite's logged boxes, a training-only target the sensor
   rule allows. It is bounded and exactly 0 when nothing closes in, so it is better conditioned than raw distance
   (C28).
3. **Only if 2 learns poorly:** one frame cannot measure expansion rate. Then add the frozen backbone's feature
   difference to the previous frame (FLARE-style latent flow; image features only, no past actions). Use
   keyframe up-weighting (Wen et al. 2021, shown in CARLA) against the copycat shortcut. CarLLaVA's null result
   for generic temporal input still applies.

### A28. Read the speed head as a decision, not a mean - **Next** (inference-only; no training)
`TargetSpeedHead.expected_speed` sends the softmax-weighted **mean** of the bins to the PID. The class's own
docstring warns that a multi-modal target's mean is "the one speed the expert never drives", and inference brings
that averaging back. P(stop) = P(8 m/s) = 0.5 gives 4 m/s: a creep toward whatever caused the stop.
Bayes decision theory gives principled replacements:
1. **Median** (optimal under absolute loss). It snaps to a mode when one mode holds >50% of the mass.
2. **tau-quantile** (optimal under pinball loss, tau = c_slow / (c_slow + c_fast)). tau < 0.5 encodes "too fast
   costs more than too slow".
3. Either one on an **HMM-filtered** posterior, `p_t ∝ softmax_t ⊙ (A^T p_{t-1})` with a sticky transition matrix.
   This smooths single-frame flicker at inference only, so it adds no network history and no copycat risk.

Order of work:
- **Offline first (no CARLA):** on held-out frames, count how often the mean lands between two modes, and compare
  mean / median / expert target speed.
- **Then evaluate** E e15 on 19 routes with the median, then tau = 0.4, with and without the filter.
- **Watch the TickRuntime count:** the hard uncertainty brake deadlocked 3 routes (S-059), and low quantiles move
  toward that. Keep the existing creep logic.

Targets both A11 (creeping into the car ahead) and A1 (creep into the obstacle).
**Measured case (S-100, literature note 8.2).** On the 19 routes, E e15 loses 34.6 DS per route: 16.7 from not
finishing and **18.0 from penalties** (7 vehicle collisions). WoR loses 32.2: **27.8 from not finishing** and 4.4 from
penalties, with 0 vehicle collisions. We sit at opposite ends of a speed-risk trade-off, and this item is the knob
that moves along it. The metric sets the cost ratio: a vehicle collision is x0.60 on the route's score, while
slowness is free until the 200 s route cap (min-speed infractions are "unused" in Bench2Drive's scorer). Halving our
penalty loss without losing completion would be worth ~+9 DS, more than the whole E-vs-WoR gap.
**Make the distribution trustworthy first (literature note 10.1, 10.3):**
- **Temperature scaling:** softmax is a Boltzmann distribution. Fit one temperature `T` on held-out NLL, since
  quantiles of an uncalibrated head mean little.
- **HL-Gauss targets (B9's Gaussian smoothing) for the speed head:** a better-calibrated distribution by
  construction. This needs retraining, so it is a second step.

**A caution from the physics:** a Bayes decision with a kinetic-energy cost (extra stopping distance
`(v_j^2 - v_i^2)/2a`) against lost progress picks the *creep* speed over a wide range of weights, because the cost is
convex in v. Energy alone does not stop the creep into parked cars (S-101). Use a committing rule
(median / tau-quantile) here, and leave gap-aware costs to A29.

### A33. Ensemble the heads over the shared frozen backbone - **Next** (eval-only; no training)
Arms A, B, D, G and A SWA share the same frozen regnety_032 at 192x512, so an ensemble costs one backbone pass plus
K small head passes. Average the target-speed probabilities (then decode as A28) and each command's waypoints.
- **Why:** run noise is a collision lottery from decisions near the stop/go boundary (S-100, S-101), and averaging
  independently trained heads smooths exactly those boundaries and improves calibration. TF++'s own agent
  (carla_garage) ensembles automatically when several `.pth` files are in its model folder.
- **Check first** whether our TF++ 80.3 baseline ran as an ensemble; that changes how the gap to it reads.
- **Measure:** 19 routes, DS, collision count, and `sigma_run` on the lottery routes (23687, 2143, 3717, 3936).
  Report it as an ensemble, not as a single arm, when comparing with WoR.
- **Caveat:** averaging waypoints averages modes (swerve vs stay). After A14's obstacle data, ensemble only the speed
  head or select a trajectory instead.

### A29. Grey-box braking: the expert's IDM applied to predicted lead-vehicle state - **Later** (needs C28/A26 heads)
PDM-Lite's longitudinal rule is IDM with published constants (A25). Predict what the rule needs, the lead
vehicle's gap `s` and closing speed `dv` (training targets from PDM-Lite's logged boxes, as C28 and A26 plan). Then
set `target = min(learned target speed, IDM(s, dv, v))`. The learned head still handles lights, junctions and
everything else. Physics extrapolates to gaps and speeds that are rare in the data, and the network does not have
to learn IDM from examples.
Evidence:
- Physics-informed car-following (PIDL-CF, TR-C 2021): IDM in the computational graph beats either part alone and
  is more data-efficient when data are sparse.
- BarrierNet (differentiable control barrier functions on network-predicted state, vision-based driving):
  obstacle-avoidance crash rate 53% -> 28%, and 3% with ground-truth state. So the gain is capped by how well
  `s` and `dv` are perceived.

Variant: train end-to-end through the IDM formula (a loss on the final target speed), so the heads learn the
accuracy braking needs. The thesis angle is that the expert is literally IDM, so the physics prior here is exact,
not approximate.

### A11. Vehicle collisions on routes we otherwise finish - **Next** (new, eval-only first)
26401 (MergerIntoSlowTrafficV2) and 27532 (BlockedIntersection) are completed at 100% route
completion in every arm (E, D x2, A SWA) but capped at 60 by one vehicle collision; WoR scores 100 on
both and has **0 vehicle collisions on all 19 routes** (its failures are timeouts). Those two routes
cost 4.2 DS, more than the whole E-vs-WoR gap. E e15 also collides on 2143, 2664, 3717, 3936, 25318.
From the eval JSONs' collision records and telemetry: who hits whom (rear-end into slow traffic,
side contact while merging, crossing traffic in the junction), at what ego speed, and whether the
target-speed head predicted a stop.
**Partly done from the JSONs (S-101, literature note section 9):**
- **Systematic collisions** (the same crash in nearly every run: 26401, 27532, 25318, 2664, 2050) are policy errors
  to fix by training.
- **Lottery collisions** (23687, 3457, 2143, 3717) cause the run noise.
- **Repeat hits come in two physical types:** ~5 m apart on the same parked car (creep, A28), and ~39 m apart on
  the same moving car on HighwayExit 23687 in 5 of 6 arms (a car alongside, likely outside the front camera, A13).
- **Ceiling:** removing vehicle collisions at unchanged completion would add +15 DS per route.

Still needed: ego speed and the target-speed head's output at each collision (telemetry). Candidate fixes depend on the answer: C37 (braking-margin hinge
loss), C28 (lead-vehicle distance as an auxiliary target, A5), or A4 (controller).

### A12. Arm H and arm I evaluations - **Arm H done; arm I finishing** (box Y, 2026-09-28)
- **Arm H** (WoR's `cnn` head on arm A's data/backbone, S-081), e15 on 19 routes: **50.4** vs E e15 65.4
  (-15.0, 95% CI [-25.4, -6.0]) and WoR 66.9 (-16.5 [-26.5, -8.0]). The controlled claim "transformer
  head > WoR head, same data and backbone" holds at e15. e20 partial 38.9 at 6/19 (no need to finish).
- **Arm I** (288x768 + colour aug 0.5, S-082): e15 58.5 at 12/19, e20 49.2 at 6/19. On the same
  routes it is level with WoR and 5-6 below arm E, so no sign it beats E. **To finish e15:** the 7 missing routes are 24795, 25318, 25975, 26401, 27532, 28035, 28198;
  copy `v2_armI_e15_x.json` back into `bench2drive_out/` on a new eval box and the guardian resumes
  from route 13. e20 needs 13 more routes. Checkpoints: E: + HF `carla_arms_20260927/carla_armI_hires_color`.
Every CARLA eval: copy `frozen_backbone.pth` with the checkpoint, grep for "Merged ... frozen
backbone" (S-072), check per-route status strings, not only the record count (S-087), and use
`scripts/eval/eval_watchdog.sh` (tree-kill) plus port clearing before each launch (S-091).

### A9. Full Bench2Drive evaluation (220 routes) - **Running: last ~50 runs on P1/P2 at Low, merge after** (S-093, S-106, S-107)
Per-route differences vs WoR have an SD of 21 (E) to 34 (A SWA) DS, so a +/-3 DS CI needs ~190-500
routes (S-090). The 220-route set is what the "beats WoR" claim needs, for the champion and the WoR
original agent, not only as a final check. Plan it now: route file, the AdditionalMaps package for
Town12/13, CARLA lanes (2-3 per box), and a time estimate from the 19-route runs (220 routes is
~11.6x the work per agent). Was C93.
**69% of the 220 routes are in Town12/13**, which neither our arms (6 towns) nor the original WoR trained on
(S-097), so A9 is mostly a generalisation test: fair for E vs WoR, not for the published table. Report
seen-town and unseen-town splits. Its 50 obstacle-type routes are the baseline and test set for A14/A16.
Report the 219-route result **two ways**: crashed routes (27515 and any auto-skipped ones, S-095) counted as
0 over 220 routes (the official Bench2Drive convention, comparable with the published table: LEAD 95.0, SimLingo
85.9, TF++ 84.2, ORION 77.7), and the paired E-vs-WoR comparison on the routes both completed.
Also run Bench2Drive's own `tools/merge_route_json.py`, then `tools/ability_benchmark.py -r merge.json`
(Merging / Overtaking / Emergency Brake / Give Way / Traffic Sign; needs exactly 220 routes, crashed ones
allowed) and `tools/efficiency_smoothness_benchmark.py`, for E and for WoR. Every recent paper reports these
abilities, none of our scripts compute them, and they show per skill where E and WoR differ, with more routes per
bucket than b2d20's route groups (S-090).

### A30. Test "beats WoR on Bench2Drive" as a fixed-route-set claim - **Next** (analysis; no box)
S-090's power analysis (+/-3 DS needs ~190-500 routes) used the SD of per-route differences (21 DS for E). That
treats routes as a random sample from all possible routes, which is the right question for "drives better in
general". For "beats WoR **on Bench2Drive-220**", the routes are fixed. The only randomness is run-to-run noise
within each route (design-based inference), so the CI depends on `sigma_run`, not on route heterogeneity.
- **Step 1 done 2026-09-28 (S-100, literature note 8.1):**
  - Run-to-run SD per route: WoR **8.6**, D e15 10.3, A SWA 10.5, A e15 18.7, B e15 20.4. Our arms are ~1.2-2.4x
    noisier than WoR.
  - ~80% of the per-route E-vs-WoR variance is run noise, so the fixed-set framing narrows the 19-route CI only a
    little. E e15 - WoR = -1.0: route-population [-9.8, +8.1], fixed-set [-8.9, +6.9].
  - At 220 routes, one run each: +/-2.5 fixed-set vs +/-2.8 route-population.
  - The earlier guess of `sigma_run` 4-8 was too optimistic.
- **Step 2, Neyman allocation (the real lever):** repeat runs in proportion to each route's `sigma_run`. On b2d20
  that means 23687 (SD 41.5), 2143 (32.0), 3717, 3936, 2286, 2050; one run is enough for 3373, 24340, 24784, 24795,
  25318, 25975, 28198. For A9, first pass one run everywhere, then repeat the routes with intermediate outcomes
  (collisions, partial completion), where runs flip.
- **Report both estimands:** fixed-set for the benchmark claim, route-population for generalisation.
- **Run noise is also a policy property.** Our policy is less repeatable than WoR, likely from decisions near a
  stop/go boundary. Log `sigma_run` as an outcome for A28 (decision layer) and A2-type weight averaging (the SWA
  hint, p = 0.25).
- **Why (S-101):** it is a collision lottery. Route completion is ~100 in almost every run, and runs differ in the
  number of vehicle collisions k (x0.60 each). For Poisson k with rate lam:
  - `E[DS] = RC*exp(-0.4*lam)`;
  - `SD[DS] = RC*sqrt(exp(-0.64*lam) - exp(-0.8*lam))`, peaking at ~29 DS for lam ~ 1.4.

  WoR barely collides, so it has no lottery. Every collision fix therefore raises the mean *and* shrinks the CI.

### A3. Automatic per-route failure classification - **Next** (tooling)
`scripts/eval/classify_failures.py` does not exist. Build it from the eval JSONs' `infractions`
(collision type, red light, stop, timeout, blocked, route deviation) and print one table per arm plus
a paired per-route comparison against WoR (the S-090 numbers came from a scratch script; fold that
in, with the paired bootstrap from A8). Feeds A11. Was C51; also decides whether C4/C50 (weather) or
C76-C82 (junction semantics) deserve reopening.

### A27. Which training frames cause our closed-loop failures? (data attribution) - **Later** (after A1's dumps)
CUPID (CoRL 2025) ranks training demos by their influence on **closed-loop return**. It drops the harmful ones
and reaches state-of-the-art diffusion policies on RoboMimic with <33% of the data, as curated.
- **Method:** our head is a deterministic regressor over frozen features, so use the gradient-similarity form
  (TracIn / TRAK with random projection). Take the states just before each failure (A1's dumps from the obstacle
  routes, A11's collisions) and rank training frames by how well their loss gradient aligns with the policy's
  output there.
- **Expected hits:** frames that teach "hold speed behind a slow car", frames where the expert reacts to an
  actor outside our camera's view (A21), and route-shifted obstacle frames (S-096) should rank high.
- **Then:** drop or down-weight them and retrain, which is cheaper than collecting data.
- **Cost:** per-sample gradients over ~400k frames of a 30M head on cached features.

Not found applied to end-to-end driving in this scan.

### A0. How much of the DS comes from vision? - **Next** (eval-only)
With a random backbone the policy still scored 50-60 DS (S-072). Run A e15 with the real backbone
vs a deliberately blanked image (route overlay only) on the 19 routes, to know how much the vision
input actually contributes on this route set.

### A24. Attention or tokenization? The same control on CARLA as B5 - **Next** (one arm; thesis claim)
Arm H's `cnn` head (`SpatialQHead`) is a 3x3 conv stack followed by global average pooling. Our transformer head
attends over the 4x4 regnety grid. Arm H's -15.0 DS (S-090) therefore shows "transformer head > WoR head", but it
cannot say whether the gain comes from **global attention** or from **keeping 16 spatial tokens**. That is the
confound "Don't flatten, tokenize!" found on Atari (B5).
- **Arm:** arm E's config with each attention layer masked to the diagonal (every token attends only to itself).
  Parameters, data and schedule stay identical, and only token mixing is removed.
- **Reading:** close to E means the gain is tokenization; close to H (50.4) means it is attention.
- **Power:** the gap it splits is ~15 DS, which 19 routes can resolve (S-090).

Use the same diagonal mask for B5's "no attention" GTrXL arm. Both tasks then answer one question with one
method (conv encoder -> spatial tokens -> transformer), which gives the thesis one claim instead of two.

### A18. Latent world model auxiliary loss (LAW) - **Next** (novel-architecture candidate)
LAW (ICLR 2025): predict the next frame's latent features from the current latents + the predicted
waypoints, supervised by the next frame's extracted latents. CARLA Town05 Long 67.9 -> 70.1 DS, NAVSIM
PDMS 77.5 -> 84.6, in a perception-free camera setting like ours. Our backbone is frozen, so the target (the
next frame's regnety 4x4 tokens) is fixed: no collapse, and the loss only shapes the transformer head. It
needs the t+k frame's features per sample (2x backbone passes, or a feature cache). It fits the WoR theme
(a learned world model) and is the base for A19.

### A17. Route/intent representation and overlay variants - **Later** (after A14)
LEAD: previous + current + next target points as an explicit token, +2.03 DS. Hecker et al. 2018:
a rendered route-planner view beats GPS coordinates. Rendered overlays work best as separate binary
channels. Variants, each one arm: (a) 3 target points as a token and as markers in the overlay; (b) a small
top-down route inset (map crop) in an image corner, giving lookahead beyond the camera's field of view;
(c) the route as a separate channel/token path instead of paint on the RGB fed to a frozen backbone.
Only after A14, because the overlay carries the S-096 mismatch today. Small effects need A9-scale routes.
(d) Distance along the route as a **colour gradient** (HAMSTER encodes time along a drawn path this way; AimBot's
plain-colour ablation lost 3 points). Today's single green line gives no distance cue beyond perspective. A25 is
the ego-state counterpart of these route variants.

### A8. Paired route bootstrap for every arm comparison - **Done as a method, keep applying**
Done for all arms vs A (S-079) and vs WoR (S-089, S-090). Every new arm (H, I) is reported as a
paired difference with a CI, against WoR and against arm A, not as a mean alone.

### A13. Camera count vs WoR - **Later** (write-up note now)
WoR drives with 4 RGB cameras (3 x 60 deg plus a narrow one, S-080), our arms with 1. State this
wherever the two are compared. C11 (3 cameras, allowed under the RGB-only rule) becomes worth an arm
only if A11 shows the collisions come from side traffic outside the front camera's view.
Evidence is mixed: LEAD's camera-only 360 deg variant reaches 91.6 DS; CarLLaVA's added rear camera cost
1.6 DS (90.40 -> 88.81).
**First concrete evidence from our runs (S-101):** on 23687 (HighwayExit), 5 of 6 arms hit the *same* moving vehicle
2-3 times, ~39 m apart. WoR (4 cameras) scores 100 on both runs. This is consistent with a car alongside, outside
our +/-55 deg front view, while the policy crosses lanes. To confirm, one telemetry run on 23687: the ego's pose at
each contact vs the other car's position.

### A4. Controller checks: lookahead and PID - **Later** (after A1/A11)
Velocity-adaptive lookahead and PID gains, as an inference-time change evaluated on the 19 routes.
Only if A1 or A11 points at the controller. Was C45, C47.
Concrete starting point from the PDM-Lite dataset-bias paper: lookahead d = 0.098 v + 0.192 (v in m/s).

### A5. Auxiliary supervision from TF++ targets - **Later**
Depth, semantics, BEV, traffic-light state, lead-vehicle distance as training-only targets (RGB
input only, per the sensor rule). Needed before any AEB-style fallback (C48) is possible without
extra sensors. Was C25, C26, C28, C31, C48; also the "TF++ BEV / depth / semantic auxiliary
supervision" line in `tried_and_ruled_out.md`.

### A34. TF++'s own depth / semantic / BEV decoders as extra inputs from the one RGB camera - **Next** (step 1 offline)
The TF++ checkpoint we already take the backbone from (`all_towns/model_0030_0.pth`, 1332 tensors) also holds
pretrained `depth_decoder`, `semantic_decoder` (7 classes), `bev_semantic_decoder` (11 classes) and CenterNet box
heads (`head.*`), all CARLA-trained, so no new perception training and no ImageNet model (sensor and backbone rules).
Different from A5: with the backbone frozen, auxiliary *targets* on its features cannot change what the policy sees;
the decoders' *outputs* (depth, drivable area, vehicles/pedestrians, lead-car boxes) fed in as extra tokens can.
**Catch:** TF++ is TransFuser. Its decoders read image features after fusion with the LiDAR branch
(`backbone.transformers`, `lidar_encoder`), and we have no LiDAR.
1. **Offline, no CARLA:** run TF++ with an all-zero LiDAR BEV on ~500 PDM-Lite frames, compare decoded depth and
   semantics with the dataset's ground truth (PDM-Lite logs both). If near its reported quality, the decoders work
   RGB-only. If not, feed pseudo-LiDAR from the decoded depth (allowed by the sensor rule) and check again.
   **First result (2026-09-29, P2, 20 frames; 500-frame run in `/workspace/tfpp_check`, MLflow `A34_tfpp_decoders_rgb_only`,
   `scripts/analysis/tfpp_decoders_rgb_only.py`):** the full TF++ model loads 1332/1332 tensors. Semantics survive a
   blank LiDAR almost untouched: pixel accuracy 99.35% -> 99.32%, mIoU 0.799 -> 0.795; vehicle IoU 0.927 -> 0.926,
   road 0.956 -> 0.955, sidewalk 0.92, road line 0.64, traffic light 0.36 -> 0.34. Depth degrades: normalised L1
   0.025 -> 0.040 overall, 0.016 -> 0.032 on the nearest 30% of pixels. So the semantic decoder can run on our camera
   alone as-is; depth works, but at about twice the near-range error.
   **500 frames / 60 routes (all towns):** mIoU 0.803 -> 0.798, vehicle IoU 0.909 -> 0.903, road 0.937 -> 0.934,
   sidewalk 0.897 -> 0.883, traffic light 0.588 -> 0.584; depth L1 0.027 -> 0.038, nearest 30% 0.018 -> 0.029. Confirmed.
2. **Arm:** depth + semantic maps pooled to the 4x4 vision grid as extra tokens (or the lead-car box as 1/TTC for
   A26). Run the decoders once per frame in the feature cache, so training cost barely moves.
**Two-frame depth (previous + current image) is weaker than it sounds:** ego motion is forward, so the stereo
baseline is ~0.3 m per 20 Hz frame at 20 km/h, and depth from motion is undefined at the focus of expansion - the image
centre, exactly where the lead car is. Two frames are better used for *motion* (closing speed; A26 step 3, the
FLARE-style feature difference), with keyframe up-weighting against the copycat shortcut (Wen et al.).

### A35. Self-prompting: draw the camera's own decoded perception into the image - **Next** (one arm + token control; after A14's WoR-head arm)
Combines A34 (TF++'s decoders work RGB-only) with the route-overlay mechanism C9 already proved. Per frame, render the
semantic decoder's vehicle/pedestrian pixels **inside the route corridor**, shaded by decoded inverse depth (near =
strong), into the camera image before the frozen backbone. Same camera, no new sensor, no perception training.
**Why it should work (own evidence):** overlays are read by this frozen backbone (C9 kept); camera-only vehicle IoU 0.90
(A34, 500 frames); collisions are our arms' largest DS loss (S-100/S-101 collision lottery); supplying missing
information just gave +15 DS (arm J, S-109).
**Novelty check (2 searches, 2026-09-29; do a full pass before claiming):** nearest are "Guiding Attention in End-to-End
Driving Models" (arXiv 2405.00242: simulator GT semantics/depth guide attention as a training loss, not a test-time
input), SUV (arXiv 2608.03084: frozen-model depth rendered as a colour map, for video generation), FROST-Drive
(arXiv 2601.03460: frozen encoder, no rendered perception). Self-prompting an RGB-only policy with its own pretrained
perception did not come up.
**Plan (~1 box-day):** decoder outputs precomputed once per frame in the feature cache; **arm L** = arm J e18 + overlay,
5 epochs (~1.5 h at arm J's 18 min/epoch); **control** = the same information as extra tokens (overlay vs token is a
finding either way); eval b2d20 + obstacle routes + an A11 collision-heavy list, paired vs J e18. The evaluator needs
one extra TF++ forward per frame. Risk: the overlay hides image detail (the token control covers it).

### A6. DAgger with the PDM-Lite expert - **Later** (large)
Roll out our policy, let PDM-Lite label the visited states, add them to training. The general
fix for compounding error and for A1-type situations the offline data never shows. Needs CARLA +
the expert running on the training box. Was C90 (and C72, C73).
**Concrete recipe (TakeAD, 2025, built on Bench2Drive):**
- PDM-Lite runs in shadow mode while our policy drives the **training** routes (not b2d20/A9).
- It takes over for 2 s when it predicts a collision or when |steer difference| > 0.2.
- About 6-8.6k samples per round. Each round: DAgger for 1 epoch, then SimPO/DPO for 10 epochs (preferred =
  expert action, rejected = the policy's top-1, beta 0.1, gamma 0.1).
- VAD base 61.82 -> 71.39 DS, saturating after round 4.

For us the preference step fits the two-hot target-speed head directly (expert speed bin preferred over our
top-1). The takeover frames also feed A11 (collisions) and A1 (obstacles) with states the offline data never
shows. Cost is harness work (PDM-Lite's privileged planner inside our leaderboard agent) plus ~1 training
epoch per round.

### A7. TF++ as a teacher - **Later**
Use TF++ outputs (path checkpoints, target speed) as extra supervision; TF++ scores 80.3 on the
same routes. Was C83 and "TF++ path checkpoints as an extra output".
Recipe: DriveAdapter (ICCV 2023) trains an adapter so student features match the frozen teacher planner's
input space, masking teacher features where the teacher breaks rules (Town05 Long 61.7 vs TCP 57.2). The
path-output part moved to A15.

### A19. Multimodal trajectory head + imagine-and-select planning - **Later** (after A1/A14, with A18)
An obstacle in the lane is bimodal (stay / swerve), and a regression head averages the modes. Options:
trajectory vocabulary + classification (VADv2 / Hydra-MDP style), or DiffusionDrive's truncated diffusion from
anchors (2 denoising steps). With A18's latent world model scoring the candidates, this becomes a planner that
imagines and selects (World4Drive / WoTE line): a thesis-level architecture that stays within the WoR idea.

### A23. Scenario-supervised action experts (DriveMoE's Action MoE) - **Later** (after A14's data)
DriveMoE (CVPR 2026, Bench2Drive): Drive-pi0 55.85 DS, Action MoE alone 67.31, with vision MoE 74.22.
- **Design:** 1 shared + 6 action experts, top-3 active; the router is trained with cross-entropy on the
  scenario/skill label, plus router noise against expert collapse.
- **Aim:** stop one head averaging across behaviour modes (swerve around an obstacle vs wait in a queue vs yield
  at a merge). It is a cheaper route to A19's problem than a trajectory vocabulary.
- **For us:** PDM-Lite archives are named by scenario, so the router labels are free. Map them onto Bench2Drive's
  5 abilities. At test time the router infers the scenario from the image.

Caveats:
- Head capacity is not binding (10M/30M/100M flat, `tried_and_ruled_out.md`), so any gain here must come from
  routing, not size.
- DriveMoE's base is a VLA. Obstacle labels exist only once arm J's Town12/13 data is in.

### A20. Closed-loop RL fine-tuning of the imitation policy - **Later** (large; thesis-level)
CaRL (CoRL 2025): PPO with one simple reward (route completion; infractions terminate the episode or scale
the reward down multiplicatively) scales to 300M CARLA samples, where shaped rewards break at large batch.
2026 methods (CRAFT, CLEAR, OPTED) learn a **residual** or KL-regularised policy around the imitation
waypoint prior. Raw2Drive (NeurIPS 2025) is the only RL method on Bench2Drive. Needs CARLA rollouts on the
training box (shares infrastructure with A6 DAgger).

### A21. Remove expert-learner asymmetry from the labels - **Later**
LEAD: PDM-Lite reacts to actors the cameras can't see and to exact velocities of other cars, which gives
non-causal or "successful but dangerous" demonstrations. Restricting the expert to camera-visible actors:
+1.37 DS. Without re-collecting data: down-weight frames where the expert's hazard/brake is caused by an
actor outside our front camera's field of view (from the logged `vehicle_affecting_id` / `speed_reduced_by_obj_*`
fields plus the camera frustum). Feeds A11.

### A10. Learning-rate sweep for the transformer head - **Later**
3e-4 was inherited from the conv head and never revisited. Arm G's grad-clip result is in (no gain,
S-079), so this is unblocked, but it is another small-effect arm: only worth it with A9's route count.

### A2. Checkpoint averaging (SWA over e11-e20) - **Done**
A SWA 62.4 vs A e15 60.6, +1.8, 95% CI [-7.2, +10.5]: no gain (S-078, S-079). Was C40.

Still open in `TODO_CARLA_EXPERIMENTS.md` but not scheduled: C2 (curvature oversampling), C6
(stationary-frame cap), C11 (3 RGB cameras, see A13), C14 (frame stacking), C19 (trajectory
history), C30 (waypoint horizon), C37 (braking-margin loss, see A11).

---

## Atari (EZ-V2 port, Breakout)

No run of our port has gone past 10k env steps; the Atari-100k benchmark itself (B4) is not started.

### B4. Scale the budget: GTrXL + uniform to 100k - **Running** (GTrXL box Z, ResNet box Y, S-093)
Atari-100k is the standard benchmark; EZ-V2's own run here gives 321.2 at 100k (paper 400.1;
curve 9.8 / 37.2 / 92.0 / 278.2 / 363.1 / ... at 10k-50k, S-058). One seed each of GTrXL and ResNet,
uniform replay, checkpoint and eval every 10k so the curve can be set against EZ-V2's. ~14 h per run
(a 10k run takes ~80-90 min on a 4070 Ti S); two runs fit on one 24 GB GPU (box T ran two streams in
17.3 GB). Needs a box without an 8-hour cap. Include `replay_latest.npz` in the backup sync: without
it a lost box restarts a 14 h run from zero (S-064, `tried_and_ruled_out.md`); check its size against
the box's bandwidth price first. Run alongside B3, not after it.

### B3. Diff our prioritized replay against EZ-V2's - **Code matched (S-104); screen next**
Priorities alpha = beta = 1 miscalibrate our reward head (S-067); uniform fixes it. Compare priority
computation, beta annealing and where importance weights are applied (value vs reward loss).
Uniform stays the default until this is done. It also decides how to read B4: at 100k, uniform may
fall behind EZ-V2's prioritized replay, and our ResNet port already differs from EZ-V2 at 10k
(15.8 vs 7.0-9.8), so it is not yet a faithful port.
**Found in the code (2026-09-28), test these first.** `EZReplay` in `train_ez_offpolicy.py` vs EZ-V2's
`ez/data/replay_buffer.py`:
1. EZ-V2 samples a batch **without replacement** (`replace=False`); ours samples **with replacement**. Under
   alpha = 1, a few high-priority reward windows can fill one batch many times over, which fits S-067's
   "reward-rich batches".
2. New data enters at EZ-V2's **current** buffer maximum, which falls as priorities are updated. Ours uses an
   **all-time** `max_prio` that never falls, so after an early value spike every new transition enters with a
   stale, very large priority.

**Implemented 2026-09-28 (S-104)**, plus a third difference: EZ-V2 clips the normalised IS weights to
[0.1, 1]; ours did not. All three apply only with `--priority-alpha > 0`; alpha 0 keeps its exact draw.
Screen: alpha = beta = 1 on the fixed code, ResNet, 10k, 2 seeds, against S058f's
uniform 15.0/15.3. IS weights scale the whole loss in both codebases (same). If priorities still hurt, the next
candidate is SpeedyZero's Priority Refresh (periodic recompute of stale priorities, reported to stabilise value
training). It matters for B4: ResNet s0 is 43.0 at 30k vs EZ-V2's own 92.0.

### B8. More updates per sample with resets (SR-SPR / BBF recipe) - **Next** (after B4's baseline curve)
The most documented sample-efficiency lever that our port lacks. BBF:
- replay ratio 8 (2 = cheap setting);
- every 40k gradient steps, shrink-and-perturb the conv layers 50% toward a random init and fully reset
  the later layers;
- n-step annealed 10 -> 3, discount 0.97 -> 0.997;
- AdamW weight decay 0.1.

BBF sits a constant +0.45 IQM above SR-SPR at every replay ratio, and its gains grow with the ratio.
EZ-V2 runs about 1.2 updates per env step with no resets. First step: replay ratio 2 plus resets of the
heads and dynamics every 40k updates, at 100k env steps. It doubles update cost, so do B10 first or alongside.
Cheaper reset variant: **Hare & Tortoise** (ICML 2024). A slow EMA copy (tortoise) of the network, with the
trained network periodically reset to it. It keeps plasticity without BBF's post-reset score drop, and the paper
reports gains on Atari-100k agents. Our target network (hard copy every 200 updates) can become the EMA tortoise.
Try it before full resets if B15 shows plasticity loss.

### B9. HL-Gauss for value, reward and value-prefix heads - **Next** (cheap)
"Stop Regressing" (ICML 2024): a Gaussian-smoothed categorical target (HL-Gauss, sigma ~0.75 bin width) beats
two-hot, MSE and C51 on Atari. It is a small change in our categorical losses, which are two-hot now, and
the reward head is where our port broke before (S-058, S-067).

### B10. Faster reanalyze (ReZero) and adaptive search - **Next** (engineering)
Reanalyze targets take ~71% of our wall-clock (`time c/t/u` 6/71/23). ReZero (2024): reuse child values
from the backward view plus periodic whole-buffer reanalyze, for less search time at equal or better
score. V-MCTS (2022): adaptive simulation budget. Seeds are the bottleneck for every Atari claim (B1), so
wall-clock is sample size.

### B2. An evaluation that can carry the claim - **Next**
Both harnesses repeat episodes: ours (10 episodes, deterministic search, 1-30 no-op starts, e.g.
[57, 57, 57, 33, 57, ...]) and EZ-V2's own (seed 0's 30 episodes gave only 3 distinct scores: 5, 6,
13; S-086). Re-evaluate every final checkpoint of both (our 12 ResNet/GTrXL 10k finals, EZ-V2 seeds
0-4) with 30+ episodes and sticky actions (p = 0.25) in **one** harness, so the comparison with
EZ-V2 is like for like.
**Protocol note (literature note 9.4):** the Atari-100k protocol (SPR, EfficientZero, BBF) evaluates **without**
sticky actions. Sticky p = 0.25 is fine for the one-harness comparison and as a robustness check, but comparisons
with EZ-V2's published 400.1 need the no-sticky protocol (more episodes, distinct no-op starts). Breakout scores turn
bimodal once the ball tunnels behind the bricks (GTrXL s0 30k: 62-92 vs 288-310). Report the tunnel rate
(P(score >= 200)) and the median next to the mean.
Analyse the result as seed x episode variance components (a hierarchical model), not as independent episodes.
Between-seed vs within-seed variance says whether the next GPU-hour should buy seeds or episodes (Neyman
allocation, the Atari counterpart of A30). At GTrXL's seed SD it is almost certainly seeds.

### B7. Official EZ-V2 seeds at 10k - **Done (5 seeds)**
`scripts/training/ezv2_10k_seeds.sh` + `ezv2_setup_blackwell.sh` (S-086). 30 episodes each with EZ-V2's
own eval: seeds 0-4 = 7.03, 15.10, 26.93, 1.30, 0.00 (mean 10.1, SD 11.1). box3's older seed 0: 9.8.
Seed 5 was at 1,650/10k when the box was destroyed (lost). EZ-V2 at 10k is as seed-dependent as
our port; our GTrXL mean 20.2 vs EZ-V2 10.1 is not yet a claim until B2 (one harness).

### B1. Seeds per trunk at 10k - **Done, don't extend**
6 seeds each: GTrXL 38.6 / 19.2 / 8.8 / 27.7 / 10.4 / 16.4 (mean 20.2, SD 11.3); ResNet 15.0 / 15.3 /
14.4 / 19.4 / 10.6 / 20.2 (mean 15.8, SD 3.5). Difference +4.4, bootstrap 95% CI [-3.6, +13.5]: not
separated. At GTrXL's seed spread, ~57 seeds per trunk would be needed for 80% power at this gap,
so more 10k seeds are not the way to the claim; B4 (100k) and B5 are. Seeds 6-7 were lost with box T
(S-085) and are not rerun.

### B5. Why does GTrXL beat ResNet? - **Later** (after B4)
Parameter-count-matched ResNet control, and a GTrXL without gating, at the budget where B4 shows a
real gap. Decides whether the thesis can attribute the gain to the transformer trunk.
Literature (2025): "Don't flatten, tokenize!" shows SoftMoE's gain comes from **tokenizing the encoder output**
(one token per spatial position), not from the experts; one scaled expert matches four. "Mind the GAP" shows
that global average pooling also beats flattening. Arms: ResNet+flatten (current), ResNet+GAP, ResNet+tokens with
a per-token MLP (no attention), GTrXL. Only the last two differ by attention.
Simplest "no attention" arm: the same GTrXL with attention masked to the diagonal (each token attends only to
itself). It has exactly the same parameters, so no count matching is needed. A24 runs the same control on CARLA.

### B6. More games - **Later** (after B4)
A few Atari-100k games beyond Breakout (e.g. Pong, MsPacman, Seaquest) with the winning config.

### B11. Multi-step contrastive consistency (TWISTER's AC-CPC) - **Later** (novel)
TWISTER (ICLR 2025): action-conditioned contrastive prediction of future latents (several steps ahead,
crop-resize augmentation) beats next-state prediction for transformer world models (1.62 mean HNS without
search). Our port uses EZ-V2's one-step SimSiam consistency per unroll step. A contrastive multi-step
version inside a search agent is untested in the literature.

### B12. Temporal latent history in the trunk (UniZero-style) - **Later** (novel)
UniZero (2024): a transformer over the latent history (KV cache) inside MuZero-style planning. On par
with or above MuZero-style agents on Atari 100k from single frames, and better on memory tasks. Our GTrXL
attends over the 36 spatial tokens of one step only. Breakout needs little memory, so this belongs with
B6 (more games).

### B13. Plasticity at longer budgets - **Later** (only if B4 stagnates late)
LayerNorm + weight decay together (Lyle et al. 2024), or SimbaV2's hyperspherical normalisation. S058e's
LayerNorm failure was under the broken prioritized replay (S-064), so it does not rule this out under
uniform replay.
Flat minima: PLASTIC (NeurIPS 2023) on Atari-100k, DrQ 0.258 IQM; +SAM (sharpness-aware minimisation) 0.325;
+reset 0.343; SAM + reset + LN + CReLU 0.421. SAM costs ~2x per update. It is the loss-surface-smoothness lever;
try it after B15, if B15 shows lost plasticity.

### B14. Object-centric input (OC-STORM / ObjectZero) - **Later** (novel, heavier)
Masks from a pretrained segmenter (SAM2/Cutie) plus ~6 labelled frames per game as extra input channels or
tokens. OC-STORM beats STORM on 18/26 games. ObjectZero (2026) puts an object-centric world model
under MCTS. It is the Atari counterpart of an input overlay; check it against the "pretrained vision only"
rule (a pretrained segmenter is allowed; training it is not).
Counter-evidence for Breakout: SAM masks added to PPO's input ("Virtual Augmented Reality", 2023) helped 4/12
games but **not Breakout or Pong**, at ~500x the training time. Don't start B14 on Breakout.

### B19. Mirror symmetry: flip Breakout and swap LEFT/RIGHT - **Later** (cheap; then a thesis angle)
Breakout is near left-right symmetric. A reflected transition with LEFT and RIGHT swapped is another valid
transition, so it is extra data for free under the 100k budget.
- **Step 1, augmentation:** in `augment`, with p = 0.5 flip the whole window (root frame and consistency
  targets), swap LEFT/RIGHT in the unrolled actions and in the policy target, and leave value and reward as they
  are. First check the symmetry numerically on real frames (walls, paddle start, flip centre after the 96x96
  resize; mirrored score digits are expected). Game-specific knowledge makes this a separate row, not
  comparable with EZ-V2's 100k numbers.
- **Step 2, thesis angle:** Equivariant MuZero proves that equivariant networks make MuZero's whole search
  equivariant (tested on ProcGen mazes, not Atari). Attention over tokens is already permutation-equivariant, so
  a **mirror-equivariant GTrXL** needs only a mirror-symmetric position encoding plus an equivariant action head.
  A CNN trunk needs mirrored filters. "Symmetry is cheap to build into token mixing" is a transformer-specific
  claim B5 could carry. Two code facts:
  - the GTrXL position table is learned and absolute (`ez_model.py`, `self.pos`), so tie it mirror-symmetric;
  - the dynamics network encodes the action as one scalar plane `a/|A|` (MuZero's convention), which orders the
    discrete actions. A LEFT/RIGHT swap is then not a linear map on the input, so step 2 needs one-hot action
    planes.
- **Only for games with a symmetry** (B6): Pong has a vertical flip with UP/DOWN swapped; many games have none.

### B15. Plasticity diagnostics on the saved checkpoints - **Next** (no box needed; before B8/B13)
Before paying for resets (B8) or normalisation fixes (B13), measure whether our runs lose plasticity at all. For
each saved checkpoint of GTrXL s0 and ResNet s0 (best, 40k, 50k/60k on E:), run a few replay batches through
the model and record:
- the dormant-neuron ratio (tau = 0.025, Sokar et al. 2023) per layer;
- the feature effective rank (srank) of the representation;
- the weight norm, and the gradient norm per layer on a fixed batch.

Rising dormancy or falling rank from 20k to 60k, alongside a flat score, would justify B8/B13. Flat metrics
point elsewhere (search, targets). Log the same metrics in `train_ez_offpolicy.py` at every eval from now on
(cheap: one batch).

### B18. GTrXL seed variance: attention entropy, then QK-norm - **Next** (step 1 needs no box, with B15)
GTrXL's 10k seed SD is 11.3 vs ResNet's 3.5 (B1: 38.6 ... 8.8; s1 scored 0.0 at upd 8004). At that spread,
~57 seeds per trunk are needed for the claim. Matching ResNet's spread would cut that roughly tenfold.
Zhai et al. (ICML 2023): transformer training instability coincides with **attention-entropy collapse**.
sigma-Reparam or QK-normalisation prevents it and makes training robust across seeds.
- **Step 1 (offline):** per-head attention entropy on the saved 10k finals. Do the good seeds (s0 38.6, s3 27.7)
  and bad ones (s2 8.8, s4 10.4) differ? Log it at every eval from now on, alongside B15's metrics.
- **Step 2 (only if step 1 shows low entropy in the bad seeds):** add QK-LayerNorm to GTrXL's attention (a few
  lines). Run 10k seeds and compare the SD, not only the mean.

Energy view: attention is a modern Hopfield network's update (Ramsauer et al. 2020). Its inverse temperature beta
decides between retrieving one pattern (entropy collapse) and averaging many. QK-norm is a way of fixing beta.

### B20. Symmetric relative position bias for GTrXL (translation + mirror) - **Later** (cheap; with B5/B19)
The learned absolute position table (`ez_model.py`, `self.pos`) breaks two symmetries a convolution trunk has for
free: translation (the ball bounces the same way anywhere) and mirror (B19). Replace it with an attention bias that
depends only on the offset `(|dx|, dy)`: 66 scalars per head on the 6x6 grid, symmetric under both.
Initialise it local (conv-like), as ConViT's gated positional attention does. ConViT reports much better sample
efficiency than DeiT in low-data regimes, and our GTrXL already starts as the ResNet (zero-initialised output),
so it would start local too. 10k screen, 2+ seeds. Report the seed SD as well (B18).
For the thesis: it tests whether GTrXL's gain needs *global, position-free* mixing or only *symmetric* mixing, next
to B5's diagonal-mask arm.

### B21. Train on the raw score through the h-transform we already have - **Later** (one flag; needs 30k+ steps)
The port trains on sign-clipped rewards (`clip_reward=True`, EZ-V2's recipe) and is scored on raw points.
- **Mismatch:** Breakout's rows are worth 1 / 4 / 7 from the bottom, so the training objective counts bricks while
  the metric weights top bricks 7x. The top rows also speed the ball up, a risk the clipped objective sees without
  the reward.
- **The fix is already built:** our value and value-prefix heads already use MuZero's invertible h-transform over
  h-space [-300, 300]. That transform was introduced (Pohlen et al. 2018) to drop reward clipping, and MuZero trains
  on raw rewards. EfficientZero / EZ-V2 / LightZero keep both, a contradiction raised in LightZero issue #239 and
  never answered or ablated there.
- **Test:** `clip_reward=False` in training only. Top rows are rarely reached at 10k, so compare at 30-50k: raw score
  and tunnel rate (B2), against the clipped run's curve.
- **Evidence is thin:** DQN-family agents find the tunnel under clipping anyway. Either result is reportable, since
  the EZ line has never ablated it.

### B24. Test-time search scaling on saved checkpoints - **Next** (eval-only; no training)
Training and evaluation both use 16 simulations (ours and EZ-V2's). Atari-100k limits environment steps, not
compute, so search depth at test time is a free, reportable knob.
- **Run:** evaluate the saved checkpoints (the 12 GTrXL/ResNet 10k finals, and the B4 s0 30k-60k states) at
  16 / 32 / 64 / 128 simulations under the B2 protocol.
- **Two readings:**
  1. A free score gain, reported as its own row.
  2. The thesis angle: a more accurate learned model gains more from deeper search, so the slope of score vs
     simulations measures model quality. If GTrXL's slope is steeper than ResNet's, the transformer's advantage is
     in the dynamics model, not only in the policy prior (B5).
- **Cost:** evaluation only, up to 8x the per-step eval time at 128 simulations.

### B22. Bound the latent: SimNorm on the EZ state - **Later** (cheap screen; with B18)
EZ-V2 turns MuZero's latent min-max normalisation off for Atari (`state_norm: False`), and our port copies this. The
latent the dynamics network feeds back into itself is bounded only by BatchNorm. Our worst failures happened in
imagined states (S-058: phantom reward at depth 1, value inflation), and GTrXL's seeds diverge (B18).
- **SimNorm (TD-MPC2):** softmax over groups of 8 channels at a fixed temperature, applied after the representation
  and after every dynamics step. The latent is then bounded and sparse by construction, and TD-MPC2's ablations
  call it essential for stability.
- **Screen:** ResNet and GTrXL, 10k, 2+ seeds. Look at the seed SD (B18) and at imagined-reward calibration
  (the S-058 probe), not only the score.

The evidence is continuous control, not Atari.

### B23. Vectorise: a sync-free, graph-captured search, then several seeds per process - **Next** (engineering)
Seeds are the binding constraint for every Atari claim (B1: ~57 per trunk at GTrXL's spread).
- **Step 1, the search.** `gumbel_mcts.py` runs `bool(active.any())`, a GPU-to-host sync, inside the selection loop,
  itself inside the 16-simulation loop. It also issues many small kernels, on the reanalyze path that is 71% of
  wall-clock (B10).
  - Loop to the known depth bound with the existing `active` mask instead of breaking early.
  - Capture each fixed-shape simulation step with CUDA graphs (`torch.cuda.graphs` or
    `torch.compile(mode="reduce-overhead")`), as DeepMind's `mctx` does with fully jitted loops.
  - Check that the policy/value outputs match bit-for-bit on a fixed seed.
- **Step 2, seeds.** K independent seeds in one process: stacked parameters via `torch.func`
  (`stack_module_state` + `vmap`) or grouped convolutions, one batched search over K x B roots, all envs in one vector
  env. PureJaxRL trains 2048 PPO agents in half the time of one PyTorch agent. Our EZ nets are small enough that
  K = 4-8 per GPU is plausible, against the 2 separate processes we fit today.
- **First:** profile one 10k run (GPU utilisation, kernel count and host syncs per update) to confirm it is
  launch-bound. This is B10's engineering half; ReZero is the algorithmic half.

### B17. Path-consistency value regulariser (GW-PCZero) - **Later** (novel for EZ-V2; after B3/B4)
GW-PCZero (NeurIPS 2023, built on EfficientZero's code): the value estimates along the search's best path
(accumulated reward + discounted value) should be equal. A loss enforces this, down-weighting uncertain nodes.
It reports **198% mean HNS vs EfficientZero's 194% at 25% of the compute**.
- **Fit:** our reanalyze already runs a search per sampled state (71% of wall-clock, B10), so the path values are a
  by-product and the loss is nearly free.
- **Novelty:** untested with Gumbel search and EZ-V2's mixed value target.
- **Before building:** read the paper's exact loss form and weights. The second-pass scan could not parse the PDF.

### B25. Gate re-closing as a plasticity reset for the GTrXL trunk - **Later** (novel; low-moderate odds)
The mixer starts as the identity (GRU gates closed, bg_init = 2; zero-initialised output). Periodically re-closing the
gates (reset the gate bias, keep all weights) would restore the trunk's plasticity without wiping learned features:
a gentler version of BBF's resets (B8) specific to gated transformers. Not found in the literature for MCTS agents.
Only after B15 shows plasticity loss; odds are lower than B20/B21 because the Atari evidence is one noisy seed.

### B26. A world model that learns how symmetric each game is (soft, detected symmetry, used in search) - **Next** (novel; step 0 eval-only)
User's concern (2026-09-29): a hard-wired mirror (B19/B20) only helps the few games that have one. So learn per game how much
symmetry to use, with near-zero downside where there is none.
**Prior work (searched 2026-09-29):** SiT (arXiv 2406.15025) already reports symmetry-invariant attention on Atari 100k, model-free
and with *fixed* symmetries, which lowers B20's novelty. Soft/learned symmetry in general: Residual Pathway Priors (2112.01388),
Augerino (2010.11882), learnable-augmentation symmetry discovery (2506.03914), partially equivariant RL under symmetry breaking
(2512.00915). Exact, known symmetry in MuZero: Equivariant MuZero (ProcGen). Not found: learned, partial symmetry inside an
MCTS latent world model, used by the search.
1. **Candidates fixed, action map free:** h-flip, v-flip, translation. The action permutation comes from ALE's action meanings
   (LEFT<->RIGHT, UP<->DOWN, ...), so it works for any game with no learning.
2. **Soft symmetry in the GTrXL trunk:** offset-only symmetric attention bias (B20) plus a **gated asymmetric residual**, gate
   closed at init and penalised (RPP). Broken symmetry opens the gate, so the model falls back to today's GTrXL.
3. **Symmetry detector:** at every eval, the model's equivariance gap on replay (dynamics: g(Ts, sigma a) vs T g(s, a); reward
   and value invariance). It sets the weight of an optional flip-consistency loss and is itself a per-game result.
4. **Use it in search (the novel twist):** at the MCTS root, also evaluate T(s), map its policy back through sigma, and average,
   weighted by the detected confidence.
**Step 0 (no training, run first, with B24):** step 4 at test time only on the saved s0 checkpoints: flip-averaging on vs off.
Gains on Breakout and losses on an asymmetric game would show the symmetry is exploitable and must be detected, not assumed.
**Games:** a clear mirror (Breakout), vertical + UP/DOWN (Pong), partial/maze (MsPacman or Boxing), a directional negative
control (e.g. RoadRunner). The detector, not these guesses, decides which is which. 30k screen, 3 seeds, then 100k.
**Risk:** gates and detector may not settle within 100k steps; the closed-gate init and step 0 bound it.

### Atari: which novel direction has the best odds (2026-09-29 review of B3-B25)
Ranking by (novel) x (probability of a clean, reportable result) x (cost), after two literature checks:
1. **B24 first, as analysis (no training):** test-time search scaling on the saved s0 states; the score-vs-simulations
   slope as a model-quality measure per trunk. Near-certain to yield a result; cheap.
2. **Superseded by B26 (SiT, arXiv 2406.15025, already does fixed symmetric attention on Atari 100k): B20 + B19 step 2 - a symmetric, position-free GTrXL** (offset-only attention bias, tied
   mirror-symmetric, conv-like local init). Transformer-specific (a CNN needs mirrored filters for the same prior),
   so it carries the thesis's trunk question (B5). ConViT's low-data evidence supports it; untested in latent
   MCTS models. Screen at 30k with 3 seeds (10k cannot resolve it: B1 needs ~57 seeds there), report the seed SD (B18).
3. **B21 raw-score training (one flag):** the EZ line clips rewards while keeping MuZero's h-transform, and nobody has
   ablated it (LightZero issue #239 unanswered). Moderate odds; either result is reportable.
Lowered: B19 step 1 (mirror augmentation) - equivariant augmentation is established outside Atari, and the EfficientZero
paper reports data augmentation gives it limited gains. Every claim needs B6 (more games) and B2 (a like-for-like eval).

### B16. Does the GTrXL trunk keep plasticity better than ResNet at higher replay ratios? - **Later** (thesis angle)
The external plasticity report (2026-09-28, see the literature note section 4) cites evidence that recurrent/gated
networks resist plasticity loss at high replay ratios (MARL, arXiv 2404.09715). Our B4 compares the two trunks
at replay ratio ~1.2. The question "trunk x replay ratio (1.2 / 2 / 4), with B15's metrics logged" ties the
trunk comparison to the most studied sample-efficiency lever. Either answer is reportable. Needs 6 runs x 2
seeds at 100k: do it only after B4's baseline curves and B15.

---

## Ops

- Rotate the Hugging Face token: it was written in plain text in `launch_boxR.sh` (box R, destroyed)
  and may be in backups on `E:`. New launchers read it from `.env` and no box has it now.
- `challenges/` is gitignored: copy it to `E:\MThesis_EXP\challenges\` at the end of every session.
- After doc/TODO changes, rebuild graphify (`/graphify --update`); it does not auto-update.
- HF push (stage 2) once checkpoints are on `E:` and results are stable. Done for arms H/I (2026-09-27).
- The HF token was put in RAM (`/dev/shm/hf_token`) on boxes U and V on 2026-09-27; both are destroyed.
- Vast containers: check `pids.max` / `cpu.max` (cgroup v2) or `cpu/cpu.cfs_quota_us` (v1) before sizing
  Ray/CARLA parallelism; `cpu_count()` reports every host CPU (S-086).
