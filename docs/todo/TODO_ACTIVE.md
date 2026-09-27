# Active TODO (created 2026-09-25)

The short list of what is worth doing next, after triaging the older TODO files on 2026-09-25.
Older items are ticked in their own files with the reason under each. Before adding anything here,
check `challenges/tried_and_ruled_out.md`: if it is there, it has been done.

Status: **Running** / **Next** / **Later**. Times are Athens.

---

## First on the next boxes (carried over from the 2026-09-25 teardown, S-076)

Where things stand (19 b2d20 routes, real backbone): A e15 60.8, B e15 53.6, **D e15 64.6**, E e20 60.6,
TF++ 80.3. Atari 10k finals: ResNet 15.0 / 15.3 / 14.4, GTrXL 38.6 / 8.8 / 27.7.

| What | Needs | Log |
|---|---|---|
| 19-route evals not finished: A SWA e11-e20, G e15, G e20, E e15, D e20 (checkpoints + `frozen_backbone.pth` are on `E:`; SWA in `live_20260925_boxB_5070_carlaeval/checkpoints/carla_armA_swa`) | an eval box (1 GPU, ~12 GB, 2 streams, ~1 h per pair) | S-076 |
| Repeat D e15 and A e15 on the 12 non-sentinel routes, then paired bootstrap (A8): is colour aug's +3.8 real? | eval box | S-075 |
| Atari 10k seeds still missing: GTrXL s1 final eval, GTrXL s4, ResNet s3, ResNet s4 | 1 GPU each, ~1.5-2 h per run | S-074, S-076 |
| Copy `frozen_backbone.pth` with every checkpoint and grep for "Merged ... frozen backbone" | - | S-072 |

---

## CARLA (the gap to TF++ is ~18 DS points, mostly the obstacle-in-lane routes)

### A1. Why the car drives into the obstacle - **Next** (highest value, eval-only)
Routes 2509, 2664, 24795 (ConstructionObstacle, ParkedObstacleTwoWays) fail in every arm and did
not improve with 8-town data that contains these scenarios (S-063). PDM-Lite's expert *leaves the
lane* to pass; our policy is conditioned on route points and a route overlay that go straight
through the obstacle. Check, from telemetry already recorded (`tel_armA_e15`, S-066) plus one
new run with waypoint/overlay dumps on these three routes:
- Do the predicted waypoints bend around the obstacle at all, or follow the route line?
- Does the target-speed head drop to 0 before the obstacle, or keep a creep speed into it?
- In the training frames of these scenarios, where are the route points relative to the expert's
  actual path? If the route runs through the obstacle while the expert swerves, the policy is being
  taught two conflicting signals.
Outcome decides the fix: route conditioning (e.g. route points from the expert's driven path in
lane-change segments), speed head, or controller (A4). Absorbs C5, C69, C78.

### A2. Checkpoint averaging (SWA over e11-e20) - **Queued**
Arm A e11-e20 averaged by `swa_avg.py` on the eval box (heads only; frozen backbone shared) and
queued for the 19 routes as `v_armA_swa`. Was C40.

### A0. How much of the DS comes from vision? - **Next** (eval-only)
With a random backbone the policy still scored 50-60 DS (S-072). Run A e15 with the real backbone
vs a deliberately blanked image (route overlay only) on the 19 routes, to know how much the vision
input actually contributes on this route set.

### A3. Automatic per-route failure classification - **Next** (tooling)
`scripts/eval/classify_failures.py` does not exist. Build it from the eval JSONs' `infractions`
(collision type, red light, stop, timeout, blocked, route deviation) and print one table per arm, so
arms are compared by failure type, not only by mean DS. Was C51; also decides whether C4/C50
(weather) or C76-C82 (junction semantics) deserve reopening.

### A4. Controller checks: lookahead and PID - **Later** (after A1)
Velocity-adaptive lookahead and PID gains, as an inference-time change evaluated on the 19 routes.
Only if A1 points at the controller. Was C45, C47.

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

### A8. Paired route bootstrap for every arm comparison - **Next** (analysis)
Arms now differ by a few DS points with one run per route; run-to-run spread on one checkpoint is
~5 points (S-059). Use `scripts/analysis/check_val_noise.py`-style paired bootstrap over the
19 routes for A vs B vs D vs E vs G, and report CIs, not just means.

### A9. Final full evaluation of the chosen checkpoint - **Later** (end of project)
The full Bench2Drive route set with repeats for the final champion and the WoR baseline. Was C93.

### A10. Learning-rate sweep for the transformer head - **Later**
3e-4 was inherited from the conv head and never revisited. Only after arm G's grad-clip result,
since both change the update size.

Still open in `TODO_CARLA_EXPERIMENTS.md` but not scheduled: C2 (curvature oversampling), C6
(stationary-frame cap), C11 (3 RGB cameras), C14 (frame stacking), C19 (trajectory history), C30
(waypoint horizon), C37 (braking-margin loss).

---

## Atari (EZ-V2 port, 10k finals: GTrXL 38.6 / 8.8 / 27.7, ResNet 15.0 / 15.3 / 14.4)

### B1. Four+ seeds per trunk at 10k - **Partly done** (see the carry-over table)
Seeds 0-3 for S058f (ResNet) and S058g (GTrXL). Report mean +/- CI per trunk and the paired
difference. This is the claim "GTrXL beats the faithful EZ-V2 network at 10k".

### B2. An evaluation that can carry the claim - **Next**
Current eval: 10 episodes, deterministic search, 1-30 no-op starts, so episodes repeat
(e.g. [57, 57, 57, 33, 57, ...]). Re-evaluate every final checkpoint with 30+ episodes and sticky
actions (p = 0.25) or sampled no-ops, and put EZ-V2's own 10k checkpoint (9.8 in its eval) through
the **same** harness so the comparison is like for like.

### B3. Diff our prioritized replay against EZ-V2's - **Next** (code reading)
Priorities alpha = beta = 1 miscalibrate our reward head (S-067); uniform fixes it. Compare priority
computation, beta annealing and where importance weights are applied (value vs reward loss).
Uniform stays the default until this is done.

### B4. Scale the budget: GTrXL + uniform to 100k - **Next after B1/B2**
Atari-100k is the standard benchmark; EZ-V2's own run here gives 321.2 at 100k (paper 400.1).
Needs a box without an 8-hour cap (a 10k run takes ~80-90 min on a 4070 Ti S; 100k is roughly 10x).
Checkpoint every few thousand steps and sync as usual.

### B5. Why does GTrXL beat ResNet? - **Later**
Parameter-count-matched ResNet control, and a GTrXL without gating, at 10k with 4 seeds. Decides
whether the thesis can attribute the gain to the transformer trunk.

### B6. More games - **Later** (after B4)
A few Atari-100k games beyond Breakout (e.g. Pong, MsPacman, Seaquest) with the winning config.

---

## Ops

- Rotate the Hugging Face token: it is written in plain text in `launch_boxR.sh` on box R.
  New launchers read it from `.env`.
- Commit the uncommitted work (colour aug flag, `challenges/`, TODO triage) when you say so.

---

## Status snapshot: 2026-09-26 18:35 Athens (before any box-destroy decision)

**Boxes:** S (`167.179.178.247:15016`, RTX PRO 4000 Blackwell 24GB) and T (`58.136.106.208:36241`,
same GPU). Both provisioned 2026-09-26 as the carry-over from the 2026-09-25 teardown (S-077).

### CARLA (box S) - all real backbone, S-072 fix in place

**Finished, 19 b2d20 routes:**
| Arm | Score | Note |
|---|---|---|
| D e15 colour aug, run 1 | 64.6 | |
| D e15 colour aug, run 2 | 64.9 | |
| D e15 colour aug, run 3 (partial) | 60.0 @ 6/19 | still running |
| **E e15 (288x768)** | **65.4** | current best, all routes completed/scored |
| A SWA (avg e11-e20) | 62.4 | |
| G e15 (grad_clip 15) | 62.2 | |
| D e20 | 61.7 | |
| G e20 | 60.9 | |
| B e15, 2nd run (12 routes) | 63.2 | |
| A e15, 2nd run (12 routes) | 56.8 | |
| E e20 | 60.6 | |

**Running:**
- WoR original agent (PCLA `wor_lb`, untuned baseline) - 8/19 routes, 47.2 so far. Process alive,
  508% CPU, just slow from sharing the box with training + 4 other eval streams. First full pass
  ETA uncertain given contention; second pass queued after.
- **Arm H** (WoR `cnn` head, matched to arm A's exact data/hyperparameters) - training, epoch 4/50,
  loss decreasing normally (0.85->0.78). This is the missing controlled baseline for "transformer
  beats WoR" (S-081).
- **Arm I** (arm E's 288x768 + colour aug 0.5, the untried combination) - queued, waiting for the
  2 active evals to free GPU headroom before starting (S-082).

**Not yet done:** WoR run 2 (repeat for a paired CI), arm H eval on 19 routes once trained,
arm I training + eval, a proper paired bootstrap across all arms with the now-larger sample.

### Atari 100k (box T)

**10k finals, 6 seeds each (up from 5 last update):**
| Trunk | s0 | s1 | s2 | s3 | s4 | s5 | mean |
|---|---|---|---|---|---|---|---|
| GTrXL | 38.6 | 19.2 | 8.8 | 27.7 | 10.4 | 16.4 | 20.2 |
| ResNet | 15.0 | 15.3 | 14.4 | 19.4 | 10.6 | 20.2 | 15.8 |

GTrXL leads on mean but with far higher seed variance; not statistically separated (S-079's CI
included zero, unchanged in direction with 2 more seeds).

**Running:** seeds 6-7 of both trunks, in progress (s6 ResNet 5k/10k, s6 GTrXL 4k/10k).

**Not started:** re-eval of every 10k checkpoint with 30+ episodes + sticky actions (B2, so our
comparison is directly comparable to EZ-V2's own 9.8-at-10k number); EZ-V2 itself run to further
seeds (needs a non-Blackwell box, PyTorch 2.0.1/Python 3.8 pin); the 100k benchmark for our port
(no run has gone past 10k yet).

### Backup state (as of this snapshot, verified by direct file listing, not just the sync loop's
self-report - see backup-verify-empty-listing-2026-09-25 memory)

- **Box T -> E:** essentially current. Only gap: `S058f_resnet_10k_s6_uniform/checkpoint_latest.pt`
  mid-write (seed 6 still training) - not a completed-result risk.
- **Box S -> E:** was significantly behind at 18:25 (arm H's 4 epochs entirely missing; D/B/E/G
  checkpoints truncated or absent) after the local sync-loop watchdog experiment (see Ops entry
  below) interrupted normal syncing. A fresh sync pass was started 18:25 and was actively
  transferring (~2.1 GB outstanding) as of this snapshot. **Do not destroy box S until a sync pass
  completes with `still_missing=0` AND a direct file-size check confirms it** (do not trust the
  loop's own "0 missing" alone, per the 2026-09-24 false-safe incident).

### Ops note: local sync-loop reliability

Local `run_in_background: true` Bash calls are the only mechanism in this harness that reliably
survives across tool calls; backgrounding with `&`/`disown` inside a single Bash call does NOT
persist (the process is gone by the next call - confirmed empirically 2026-09-26). A local
watchdog script meant to auto-restart the sync loop on crash used `pgrep`, which does not exist
in this machine's Git Bash - every 60s it wrongly concluded the loop was dead and restarted it,
likely repeatedly killing/replacing an in-progress sync. Reverted to plain `run_in_background`
loops with a longer interval (5 min) instead of any local auto-restart layer. If a sync loop's
task shows no output for several minutes, verify with a direct box-side `find` + local `find`
size comparison before concluding anything, rather than assuming the loop's status line.

### Update 18:40 Athens: network throughput degraded, sync running but slow

Diagnosed a real network problem (not box contention): a clean 200MB test file transferred at
~80-95KB/s to BOTH box S and box T via scp (box T had synced fine for hours earlier today), while
a plain HTTP download from an unrelated host got ~420KB/s and ping to 8.8.8.8 was 223ms (high).
This points at degraded conditions on this machine's connection right now, not something fixable
by changing box-side load. Paused (SIGSTOP, not killed) and resumed the two CARLA eval processes
on box S while testing - confirmed CPU/IO contention was NOT the bottleneck, so they're back
running normally.

**Current action:** both sync loops restarted as persistent background tasks at a 3-min retry
interval, running continuously rather than waiting for a fast pass. Given current throughput,
box S has ~2.3 GB still outstanding (arm H's growing training checkpoints, D/B/E/G eval
checkpoints) - at ~90KB/s that is several hours, not minutes, if the network stays this slow.
**Do not treat "started" as "safe" - verify actual bytes-on-E: match bytes-on-box before any
box-destroy decision**, exactly as the 2026-09-24 false-safe incident warned against.

---

## Full status update: 2026-09-26 ~19:05 Athens

### Backup: logs/results prioritized, checkpoints deferred (user decision)

Given the degraded network throughput (see 18:40 entry above), the user chose to prioritize logs
and eval results over checkpoint weights for now. Two separate sync loops on box S:
- **Logs/results loop** (2-min interval, `*.log *.sh bench2drive_out guardian_logs telemetry
  MThesis/mlruns`): **caught up** as of ~18:55 - 26.9 MB of ~27.9 MB on E:, the gap being normal
  in-flight appends from running evals. This is the loop kept running.
- **Checkpoint loop** (5-min interval, finished arms A/B/D/E/G only, excluding in-progress H/I):
  stopped per user instruction ("lets do something else... logs only" / "ok lets do something
  else") - **not currently running**. ~2.18 GB of scored-arm checkpoints still outstanding on E:
  if/when resumed. Box T's loop (Atari) was untouched throughout and remains essentially current.

**Consequence:** if box S is destroyed right now, the actual trained `.pth` weight files for
arms A/B/D/E/G (all now-scored, including the current best E e15 at 65.4 and D e15 at 64.6-64.9)
would be **lost** and would require full retraining to reproduce (dataset download + ~1-2h/arm).
The scores/results themselves (what's in the challenges log, S-073 through S-082) are safe either
way once logs finish syncing - only the ability to re-evaluate those exact checkpoints on more
routes, average/ensemble them, or push them to Hugging Face would be lost.

### CARLA (box S) - scores as of this snapshot, 19 b2d20 routes, real backbone (S-072 fix)

| Arm | Score | Status |
|---|---|---|
| **E e15 (288x768)** | **65.4** | done - current best |
| D e15 colour aug, run 1 | 64.6 | done |
| D e15 colour aug, run 2 | 64.9 | done |
| D e15 colour aug, run 3 | 55.5 @ 8/19 so far | running - lower than runs 1-2, watch for variance |
| B e15, 2nd run (12 routes) | 63.2 | done |
| A SWA (avg e11-e20) | 62.4 | done |
| G e15 (grad_clip 15) | 62.2 | done |
| D e20 | 61.7 | done |
| G e20 | 60.9 | done |
| E e20 | 60.6 | done |
| A e15, 2nd run (12 routes) | 56.8 | done |
| **WoR original agent** (PCLA `wor_lb`, untuned) | 47.2 @ 8/19 so far | running, slow (box load) |

Arm H (WoR `cnn` head, matched to arm A) training: **epoch 8/50**, loss 0.78->0.56, healthy.
Arm I (288x768 + colour aug combo, S-082) still queued behind the 2 active evals.

### Atari 100k (box T) - 10k finals, 6 seeds each

| Trunk | s0 | s1 | s2 | s3 | s4 | s5 | mean |
|---|---|---|---|---|---|---|---|
| GTrXL | 38.6 | 19.2 | 8.8 | 27.7 | 10.4 | 16.4 | 20.2 |
| ResNet | 15.0 | 15.3 | 14.4 | 19.4 | 10.6 | 20.2 | 15.8 |

Seeds 6-7 running (ResNet s6 @ 7k/10k, GTrXL s6 @ 6k/10k). Not statistically separated (S-079).

### Network diagnosis (18:40, still unresolved)

Measured ~80-95 KB/s scp throughput to BOTH boxes on a clean test file (vs box T syncing fine at
normal speed earlier today), while an unrelated HTTP download got ~420 KB/s and ping to 8.8.8.8
was 223ms. Ruled out box-side contention as the cause (paused all heavy processes on box S,
throughput did not improve). This looks like a local/ISP-side degradation, not something fixable
by changing what runs on the remote boxes. Not re-tested since; may have recovered.

### What remains, in priority order

1. **Decide when/whether to resume the checkpoint backup** for arms A/B/D/E/G (~2.18 GB) - the
   real data-loss risk if boxes are destroyed before this runs.
2. WoR run 2 (repeat for a paired CI vs our best arms).
3. Arm H eval once its 50 epochs finish (or an earlier epoch, screen protocol is e15/e20).
4. Arm I: train once GPU frees up, then evaluate.
5. D e15 run 3: finish and fold into the paired-bootstrap comparison (TODO_ACTIVE A8).
6. Atari: finish seeds 6-7, then B2 (re-eval with 30+ episodes/sticky actions for a fair EZ-V2
   comparison), then the actual 100k benchmark run (none done yet on our port).
7. HF push (stage 2) once checkpoints are safely on E: and results are stable.
