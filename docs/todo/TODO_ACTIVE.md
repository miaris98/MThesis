# Active TODO (created 2026-09-25, rewritten 2026-09-27)

The short list of what is worth doing next. Older items are ticked in their own files with the reason
under each; fully closed TODO files are in `docs/archive/`. Before adding anything here, check
`challenges/tried_and_ruled_out.md`: if it is there, it has been done.

Status: **Running** / **Next** / **Later** / **Done**. Times are Athens. Item numbers are cited from
the challenges log, so they are never reused or renumbered; new items get the next free number.
The status block below is the only snapshot in this file: replace it, don't append new ones
(history goes to `challenges/`).

---

## Status: 2026-09-27 15:45 Athens

**Goal check (S-089, S-090):** the original WoR agent scores **67.8 / 66.0** (2 runs) on the 19 b2d20
routes. Best arm **E e15 65.4**, -1.5 vs WoR, 95% CI [-10.5, +7.8]: **tied, goal not met.** TF++ 80.3.

| Box | Job | State |
|---|---|---|
| V (RTX PRO 4000, CARLA) | Arm I training (288x768 + colour aug, 20-epoch screen) | epoch 7/20 at 15:03; e15 ~17:40, e20 ~19:15, then 19-route evals |
| X2 (RTX 3090, CARLA eval) | Arm H e15 on the 19 routes (valid rerun after S-087) | 11/19, 39.0 so far; H e20 after |
| W (RTX PRO 4000, EZ-V2) | Official EZ-V2 seeds 0-4 to 10k, 30-episode eval each | seed 0 7.03; seed 1 evaluating |

Backups: all three boxes sync to `E:\MThesis_EXP\live_20260927_box*` every 5 min.

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

### A11. Vehicle collisions on routes we otherwise finish - **Next** (new, eval-only first)
26401 (MergerIntoSlowTrafficV2) and 27532 (BlockedIntersection) are completed at 100% route
completion in every arm (E, D x2, A SWA) but capped at 60 by one vehicle collision; WoR scores 100 on
both and has **0 vehicle collisions on all 19 routes** (its failures are timeouts). Those two routes
cost 4.2 DS, more than the whole E-vs-WoR gap. E e15 also collides on 2143, 2664, 3717, 3936, 25318.
From the eval JSONs' collision records and telemetry: who hits whom (rear-end into slow traffic,
side contact while merging, crossing traffic in the junction), at what ego speed, and whether the
target-speed head predicted a stop. Candidate fixes depend on the answer: C37 (braking-margin hinge
loss), C28 (lead-vehicle distance as an auxiliary target, A5), or A4 (controller).

### A12. Arm H and arm I evaluations - **Running**
- **Arm H** (WoR's `cnn` head on arm A's data/backbone, S-081): the controlled "transformer head vs
  WoR head" comparison. e15 at 11/19 routes: 39.0 vs E e15 61.5 and WoR 57.5 on the same routes
  (S-090). Finish e15 and e20 on 19 routes, then paired bootstrap vs arm A and E. This claim can
  stand even if the comparison with the original WoR stays tied.
- **Arm I** (288x768 + colour aug 0.5, S-082): training on box V; 19-route evals at e15 and e20.
Every CARLA eval: copy `frozen_backbone.pth` with the checkpoint, grep for "Merged ... frozen
backbone" (S-072), and check per-route status strings, not only the record count (S-087).

### A9. Full Bench2Drive evaluation (220 routes) - **Next** (moved up from end of project)
Per-route differences vs WoR have an SD of 21 (E) to 34 (A SWA) DS, so a +/-3 DS CI needs ~190-500
routes (S-090). The 220-route set is what the "beats WoR" claim needs, for the champion and the WoR
original agent, not only as a final check. Plan it now: route file, the AdditionalMaps package for
Town12/13, CARLA lanes (2-3 per box), and a time estimate from the 19-route runs (220 routes is
~11.6x the work per agent). Was C93.

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

### A8. Paired route bootstrap for every arm comparison - **Done as a method, keep applying**
Done for all arms vs A (S-079) and vs WoR (S-089, S-090). Every new arm (H, I) is reported as a
paired difference with a CI, against WoR and against arm A, not as a mean alone.

### A13. Camera count vs WoR - **Later** (write-up note now)
WoR drives with 4 RGB cameras (3 x 60 deg plus a narrow one, S-080), our arms with 1. State this
wherever the two are compared. C11 (3 cameras, allowed under the RGB-only rule) becomes worth an arm
only if A11 shows the collisions come from side traffic outside the front camera's view.

### A4. Controller checks: lookahead and PID - **Later** (after A1/A11)
Velocity-adaptive lookahead and PID gains, as an inference-time change evaluated on the 19 routes.
Only if A1 or A11 points at the controller. Was C45, C47.

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

### B4. Scale the budget: GTrXL + uniform to 100k - **Next, start now** (was "after B1/B2")
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

### B2. An evaluation that can carry the claim - **Next**
Both harnesses repeat episodes: ours (10 episodes, deterministic search, 1-30 no-op starts, e.g.
[57, 57, 57, 33, 57, ...]) and EZ-V2's own (seed 0's 30 episodes gave only 3 distinct scores: 5, 6,
13; S-086). Re-evaluate every final checkpoint of both (our 12 ResNet/GTrXL 10k finals, EZ-V2 seeds
0-4) with 30+ episodes and sticky actions (p = 0.25) in **one** harness, so the comparison with
EZ-V2 is like for like.

### B7. Official EZ-V2 seeds 0-4 at 10k - **Running** (box W)
`ezv2_10k_seeds.sh` with the Blackwell environment port (S-086): seed 0 7.03 +/- 0.60 SE over 30
episodes (box3's older seed 0: 9.8 over 10); seed 1 evaluating. The "Aborted" line after each seed is
the script stopping the trainer at 10k, not a crash. Gives EZ-V2 a 10k distribution for B2.

### B1. Seeds per trunk at 10k - **Done, don't extend**
6 seeds each: GTrXL 38.6 / 19.2 / 8.8 / 27.7 / 10.4 / 16.4 (mean 20.2, SD 11.3); ResNet 15.0 / 15.3 /
14.4 / 19.4 / 10.6 / 20.2 (mean 15.8, SD 3.5). Difference +4.4, bootstrap 95% CI [-3.6, +13.5]: not
separated. At GTrXL's seed spread, ~57 seeds per trunk would be needed for 80% power at this gap,
so more 10k seeds are not the way to the claim; B4 (100k) and B5 are. Seeds 6-7 were lost with box T
(S-085) and are not rerun.

### B5. Why does GTrXL beat ResNet? - **Later** (after B4)
Parameter-count-matched ResNet control, and a GTrXL without gating, at the budget where B4 shows a
real gap. Decides whether the thesis can attribute the gain to the transformer trunk.

### B6. More games - **Later** (after B4)
A few Atari-100k games beyond Breakout (e.g. Pong, MsPacman, Seaquest) with the winning config.

---

## Ops

- Rotate the Hugging Face token: it was written in plain text in `launch_boxR.sh` (box R, destroyed)
  and may be in backups on `E:`. New launchers read it from `.env` and no box has it now.
- `challenges/` is gitignored: copy it to `E:\MThesis_EXP\challenges\` at the end of every session.
- After doc/TODO changes, rebuild graphify (`/graphify --update`); it does not auto-update.
- HF push (stage 2) once checkpoints are on `E:` and results are stable.
