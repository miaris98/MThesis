# CARLA iteration protocol: screen short, confirm long

**Status**: adopted 2026-09-23. Applies to every WoR/Qwen driving-policy change until a candidate beats the current champion on the full suite.

## Why this exists

One full cycle today is 50 offline epochs (~3.7 min/epoch with the frozen-backbone cache, ~3 h) followed by the 20-route Bench2Drive suite (~1.5 h with CARLA crash recovery). That is too slow to compare ideas. It also selected the wrong thing: training is offline, so no driving score exists during training, and validation error kept improving on the training set while closed-loop driving got worse.

| Checkpoint (8-town run, 2026-09-23) | Val lat. err | 20-route mean DS |
|---|---|---|
| Epoch 20 (val-best) | 3.2 cm | **56.6** (19/20 routes; 27515 crashes CARLA on load) |
| Epoch 50 | 3.9 cm | 47.4 (16/20 at time of writing) |

References on the same 20 routes: TF++ 80.33, previous Qwen 27.05, CNN 26.23 (`E:\MThesis_EXP\bench2drive_eval\run_20260916_final`).

## The two tiers

**Final candidates are always trained for 50 epochs** (the setting the reference papers use). Short runs are only for screening, and a screened idea is promoted by *resuming* its run to 50 epochs, never by retraining from scratch. This is the 4-gate protocol in `.agents/rules/progressive_4gate_scaling.md` applied to CARLA.

| Tier | Training | Evaluation | Wall clock | Decides |
|---|---|---|---|---|
| **Screen** (Gate 3) | `--epochs 50 --stop_epoch 20 --save_freq 5`: the first 20 epochs of the 50-epoch run, LR warm-up/cosine schedule computed for 50, so resuming is exact (`wor_trainer.py` fast-forwards the schedule on `--resume_from`) | Sentinel set (below) on epochs 10, 15, 20 | ~75 min train + ~20 min per checkpoint (3 arms in parallel) | Is the idea worth 30 more epochs? |
| **Confirm** (Gate 4) | Same command without `--stop_epoch`, plus `--resume_from model_epoch_020.pth` | Full 20-route b2d20 suite on the best closed-loop checkpoints | ~2 h train + ~1.5 h eval | Is it the new champion? |

Rules:
- **Checkpoint choice is by closed-loop driving score, measured after training on saved checkpoints.** Validation error is logged and reported but never used to pick a checkpoint.
- Compare within a tier only (screen vs screen on the same sentinel routes, full vs full on b2d20).
- A route that ends in a harness failure (CARLA segfault, "Failed - Simulation") is re-run and merged with `scripts/eval/merge_leaderboard_results.py`; it is never averaged in as 0. Policy outcomes ("TickRuntime", "Agent got blocked") are real results and always count.

## Sentinel route set (screening tier)

Chosen from the epoch-20/50 diagnosis to cover each failure mode plus two controls. All from `bench2drive220.xml`, so results stay comparable with b2d20.

| Route | Town | Seen in training? | Why it is in the set (epoch-20 result) |
|---|---|---|---|
| 2664 | Town12 | no | vehicle collision, then stall (TickRuntime, RC 38%) |
| 3457 | Town13 | no | stall (TickRuntime, RC 36%) |
| 28198 | Town15 | no | stall with **no** collision (RC 16%) |
| 2509 | Town12 | no | 4 layout collisions, DS 17.9 |
| 24795 | Town04 | **yes** | stall + 5 layout collisions in a seen town (epoch 50: DS 6.4) |
| 2286 | Town12 | no | control: DS 100 |
| 24784 | Town02 | **yes** | control: DS 100 (epoch 50) |

Run as three arms (`GPU=1 ARMS="s1:... s2:... s3:..." scripts/eval/launch_b2d20.sh run`), ~20 min per checkpoint.

## What the current champion gets wrong (epoch 20, 19 routes)

1. **Vehicle collisions are the biggest score loss.** Most *completed* routes carry 1-2 `collisions_vehicle` (x0.6 each), capping them at 36-60 DS despite 100% route completion.
2. **Stalls ("TickRuntime") mostly follow a collision.** Stalled routes reach 32-38% completion in 200 s of game time, usually right after a vehicle collision; 28198 (Town15) stalls without one.
3. **Chronically slow.** 19-21 `min_speed_infractions` on every route, including DS-100 ones. Not penalised by Bench2Drive's score, but it means the policy under-drives relative to traffic.
4. **Not only unseen towns.** 14/20 b2d20 routes are in towns absent from training (06, 07, 11, 12, 13, 15), but seen-town routes also fail (24795, Town04).

## Known confounds to fix before the next screen

- **The "8-town" run changed datasets at the epoch-15 resume**: epochs 1-15 used 883,990/78,059 train/val frames from 5,134 route groups, epochs 16-50 used 230,468/20,477 from 1,730 route groups (6 towns). Validation numbers before and after epoch 16 are not comparable. Every screening run must train on one fixed dataset from epoch 1.
- **The training set lived only on the destroyed 4090 box.** It must be re-downloaded (`PDM_Lite_Carla_LB2`, Hugging Face) to the training box, and its town/route/frame counts recorded here before the first screen.

## Candidate screens, in priority order

1. **Baseline re-screen on the fixed dataset**: the current recipe, 20 epochs, so every later idea has a same-tier reference.
2. **Collision avoidance signal**: the policy has no explicit notion of other vehicles beyond pixels. Options within the RGB-only rule: auxiliary supervision targets from the recorded `boxes`/BEV (supervision only, not inputs), or weighting hazard frames (`vehicle_hazard`, `junction`) in the waypoint loss.
3. **Speed/stall behaviour**: check how predicted target speed maps to throttle at inference (`bench2drive_agent.py` -> `WorldOnRailsAgent`), and whether the 5 waypoints at the logged frame spacing give enough lookahead to recover after braking.
4. **Unseen-town coverage**: add data from towns 06/07/11-15 if the dataset has it.
