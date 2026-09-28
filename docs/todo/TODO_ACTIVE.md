# Active TODO (created 2026-09-25, rewritten 2026-09-27)

The short list of what is worth doing next. Older items are ticked in their own files with the reason
under each; fully closed TODO files are in `docs/archive/`. Before adding anything here, check
`challenges/tried_and_ruled_out.md`: if it is there, it has been done.

Status: **Running** / **Next** / **Later** / **Done**. Times are Athens. Item numbers are cited from
the challenges log, so they are never reused or renumbered; new items get the next free number.
The status block below is the only snapshot in this file: replace it, don't append new ones
(history goes to `challenges/`).

---

## Status: 2026-09-28 11:55 Athens (S-098)

| Box | Job | State |
|---|---|---|
| Y (2x RTX 3090) | Arm I e20 (19/19 done, 63.9 DS), e15 (18/19 done), A9 chunks 2, 3, 9 on 4 lanes | running |
| Y GPU 1 | B4: ResNet + uniform, 100k seed 0 (`S058i_resnet_100k_s0_uniform`) | running (~17k/100k, ETA ~22:45) |
| Z (RTX 3080) | B4: GTrXL + uniform, 100k seed 0 (`S058h_gtrxl_100k_s0_uniform`) + A9 E chunk 7 | running (~17k/100k, ETA ~23:10) |
| AA (A10) | A9 chunks 4, 5, 6, 8, 10 (E e15) on 5 lanes | running (all chunks >70% done) |
| AB (4x 2080 Ti 22GB) | GPU 2: B4 GTrXL seed 1 (`S058h_s1`), GPU 3: B4 ResNet seed 1 (`S058i_s1`) | running (launched 11:53) |
| AB (4x 2080 Ti 22GB) | GPU 0 & 1: CARLA A9 Chunk 1 (E e15) + WoR baseline chunks on 4 lanes | setting up CARLA, lanes queued |

A9 = full Bench2Drive (219 routes) for arm E e15 and the original WoR. Arm E e15 ETA ~13:30 (accelerated by AB Chunk 1); WoR baseline ETA ~15:30.
**Literature scan (2026-09-28):** new items A14-A21 and B8-B14, from `docs/design/literature_scan_2026-09-28.md`.
**Top CARLA lead (S-096, S-097):** our 6-town training set has **no** obstacle scenarios (they are all in Town12/13), so
arms A-I never saw one. The one run that had them (8-town, S-063) fed PDM-Lite's obstacle-shifted `route` as input,
while evaluation feeds the unshifted plan. The fix needs both: obstacle data **and** `route_original` (A14), then
moderate oversampling (A16).
Backups: `E:\MThesis_EXP\live_20260928_box{Y,Z,AA}_*` every 5 min, replay buffers every 3 h.
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

### A14. Obstacle data + the unshifted route (`route_original`) - **Next, first** (S-096, S-097)
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

### A11. Vehicle collisions on routes we otherwise finish - **Next** (new, eval-only first)
26401 (MergerIntoSlowTrafficV2) and 27532 (BlockedIntersection) are completed at 100% route
completion in every arm (E, D x2, A SWA) but capped at 60 by one vehicle collision; WoR scores 100 on
both and has **0 vehicle collisions on all 19 routes** (its failures are timeouts). Those two routes
cost 4.2 DS, more than the whole E-vs-WoR gap. E e15 also collides on 2143, 2664, 3717, 3936, 25318.
From the eval JSONs' collision records and telemetry: who hits whom (rear-end into slow traffic,
side contact while merging, crossing traffic in the junction), at what ego speed, and whether the
target-speed head predicted a stop. Candidate fixes depend on the answer: C37 (braking-margin hinge
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

### A9. Full Bench2Drive evaluation (220 routes) - **Running** (boxes Y+Z, 5 lanes, S-093)
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

### A3. Automatic per-route failure classification - **Next** (tooling)
`scripts/eval/classify_failures.py` does not exist. Build it from the eval JSONs' `infractions`
(collision type, red light, stop, timeout, blocked, route deviation) and print one table per arm plus
a paired per-route comparison against WoR (the S-090 numbers came from a scratch script; fold that
in, with the paired bootstrap from A8). Feeds A11. Was C51; also decides whether C4/C50 (weather) or
C76-C82 (junction semantics) deserve reopening.

### A0. How much of the DS comes from vision? - **Next** (eval-only)
With a random backbone the policy still scored 50-60 DS (S-072). Run A e15 with the real backbone
vs a deliberately blanked image (route overlay only) on the 19 routes, to know how much the vision
input actually contributes on this route set.

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

### A8. Paired route bootstrap for every arm comparison - **Done as a method, keep applying**
Done for all arms vs A (S-079) and vs WoR (S-089, S-090). Every new arm (H, I) is reported as a
paired difference with a CI, against WoR and against arm A, not as a mean alone.

### A13. Camera count vs WoR - **Later** (write-up note now)
WoR drives with 4 RGB cameras (3 x 60 deg plus a narrow one, S-080), our arms with 1. State this
wherever the two are compared. C11 (3 cameras, allowed under the RGB-only rule) becomes worth an arm
only if A11 shows the collisions come from side traffic outside the front camera's view.
Evidence is mixed: LEAD's camera-only 360 deg variant reaches 91.6 DS; CarLLaVA's added rear camera cost
1.6 DS (90.40 -> 88.81).

### A4. Controller checks: lookahead and PID - **Later** (after A1/A11)
Velocity-adaptive lookahead and PID gains, as an inference-time change evaluated on the 19 routes.
Only if A1 or A11 points at the controller. Was C45, C47.
Concrete starting point from the PDM-Lite dataset-bias paper: lookahead d = 0.098 v + 0.192 (v in m/s).

### A5. Auxiliary supervision from TF++ targets - **Later**
Depth, semantics, BEV, traffic-light state, lead-vehicle distance as training-only targets (RGB
input only, per the sensor rule). Needed before any AEB-style fallback (C48) is possible without
extra sensors. Was C25, C26, C28, C31, C48; also the "TF++ BEV / depth / semantic auxiliary
supervision" line in `tried_and_ruled_out.md`.

### A6. DAgger with the PDM-Lite expert - **Later** (large)
Roll out our policy, let PDM-Lite label the visited states, add them to training. The general
fix for compounding error and for A1-type situations the offline data never shows. Needs CARLA +
the expert running on the training box. Was C90 (and C72, C73).

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

### B3. Diff our prioritized replay against EZ-V2's - **Next** (code reading)
Priorities alpha = beta = 1 miscalibrate our reward head (S-067); uniform fixes it. Compare priority
computation, beta annealing and where importance weights are applied (value vs reward loss).
Uniform stays the default until this is done. It also decides how to read B4: at 100k, uniform may
fall behind EZ-V2's prioritized replay, and our ResNet port already differs from EZ-V2 at 10k
(15.8 vs 7.0-9.8), so it is not yet a faithful port.

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

### B14. Object-centric input (OC-STORM / ObjectZero) - **Later** (novel, heavier)
Masks from a pretrained segmenter (SAM2/Cutie) plus ~6 labelled frames per game as extra input channels or
tokens. OC-STORM beats STORM on 18/26 games. ObjectZero (2026) puts an object-centric world model
under MCTS. It is the Atari counterpart of an input overlay; check it against the "pretrained vision only"
rule (a pretrained segmenter is allowed; training it is not).

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
