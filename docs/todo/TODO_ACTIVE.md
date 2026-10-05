# Active TODO (created 2026-09-25, rewritten 2026-09-27)

The short list of what is worth doing next. Older items are ticked in their own files with the reason
under each; fully closed TODO files are in `docs/archive/`. Before adding anything here, check
`challenges/tried_and_ruled_out.md`: if it is there, it has been done.

Status: **Running** / **Next** / **Later** / **Done**. Times are Athens. Item numbers are cited from
the challenges log, so they are never reused or renumbered; new items get the next free number.
The status block below is the only snapshot in this file: replace it, don't append new ones
(history goes to `challenges/`).

---

## Status: 2026-10-04 evening Athens (S-117 to S-120) - **all three boxes are gone (box 3 and box 2 destroyed after verified saves; box 1 idle, its last verified sync 16:36 with 391 files identical); no box is running; nothing was pushed to HF today**

| Item | Result |
|---|---|
| **CARLA, 220 routes (A9 final, S-117)** | official DS: E e15 63.4, arm J seeds 0-3 58.8-61.9 (pooled 59.4), original WoR 49.4. Paired vs WoR: E +13.7 [+8.6, +18.8], J +10.4 [+5.7, +15.1], every J seed +9.8 to +11.9 (CIs exclude 0). **J - E = -2.6 [-5.9, +0.7]: J is not an overall improvement over E.** |
| **Where the score goes (S-117)** | J: 20.1 of 36.9 lost points per run are vehicle collisions (47% of runs, WoR 22%). J - E: lead-vehicle routes **-15.2 [-21.5, -9.0]**, obstacle routes **+7.2 [+0.7, +13.6]**, other -1.0. Against WoR the advantage is entirely on the 102 routes that are neither (+22.8 [+16.6, +28.9]); lead-vehicle -6.7 and obstacle +2.5 are ties. Abilities (success %, J / E / WoR): overtaking 16-18 / 5.7 / 11.7, merging 26-39 / 44.9 / 21.8, emergency brake 25-38 / 32 / 30 |
| **What has worked (ledger, S-117)** | `docs/design/what_has_worked_2026-10-04.md`: confirmed wins = route conditioning, transformer head over WoR's conv head (+15), obstacle data + `route_original`; recovery camera, colour aug, 288x768 not established; vision lightly used (no-vision MLP 0.7558 vs 0.5793; random-backbone evals 50-60 DS); 73.5% of the waypoint objective is speed x dt; the field's best recipe (LEAD) published its state-aligned dataset |
| **Headroom analysis (S-120)** | from the existing records: **J's vehicle collisions are 80% systematic** (>= 3 of 4 seeds collide; 20% lottery) -> A43 demoted; E and J **drive straight into stopped vehicles** on obstacle routes (0.70 / 0.86 events per run, on the route line; WoR 0.22) and J clips cones while passing; **32% of J's colliding runs hit again (3.5 points per run)** -> A50; obstacle families are the **universal failure** (21 of the 29 routes where E, J and WoR all score < 50); E is flat in night / rain / fog while WoR loses 13 in fog (our advantage +22 in fog vs +8, change +13.7 [+1.0, +26.8]); **220 routes resolve ~5 DS at one run per arm (7 on the 102 lead + obstacle routes)** -> A52 |
| **Research deep dive (S-122, S-123)** | `docs/design/policy_options_datasets_timeseries_2026-10-04.md`: **SimLingo-Data** has our camera (1024x512, FOV 110) and ships per-frame **counterfactual crash labels** (Dreamer: accelerating crashes in 60.5% of frames of one archive) and a hazard-bucket index (A56); Bench2Drive's and TaCarla's cameras differ from the frozen TF++ camera; Fail2Drive adds 200 paired Town13 routes (A58); BevAD / AlignDrive: path -> speed cascade, space-to-depth tokens + masking, diffusion scales with data (A15, A54, A55); time-series lens: scale-free loss (A53), newsvendor decode (A57), direct multi-step heads and events for Atari (B45, B46); lookback, foundation models and SSM trunks are not supported |
| **Speed plan (S-125)** | measured lane time on the six CARLA boxes: **route 47% / per-route overhead 29% / job start-up 9% / hand-over 14%** (186 lane-hours, 2.7 routes per job); the **pooled 4x4-token cache is exact** for the J family (the head pools before any learnable layer; no random augmentation upstream) -> A59; **noise = 74% evaluation-only, 26% training seed** -> repeat runs help, finalists need 3 trained seeds; a **lead + obstacle screen** (~100 routes, 1.1 box-h) resolves 3.0 DS overall-equivalent vs 4.3 for a full run (A52); the month is **implementation-bound, not box- or money-bound**; compressed plan below: **~4 weeks P50 / 5.5 P90, ~$115-175 with reserve for all 30 items** (needs parallel implementation) |
| **New items** | **A36-A60 (CARLA) and B37-B46 (Atari)** with tiers and a first-session order (block below); A15 and A21 promoted to Next; A43 demoted (S-120); A48 demoted (S-122) |
| **B23 step 1b (graphed search in the trainer, S-118)** | **identical to eager** at batch 32 (1,600 updates) and at batch 256 on a 3090 (800 updates): all 298 model tensors, all target tensors, the whole replay buffer (acting policies and root values included); 0 fallbacks (0/200 acting, 0/800 reanalyze). |
| **B36 pilot (S-118): GTrXL 30k, mixer wd 0, seeds 0-3, plus two non-zero-output-init runs** | s2 finished: 30k **129.8** (25k 120.0, 20k 64.4); s0 25k 24.8 (20k 51.6); s1 25k 35.1 (20k 53.7); s3 20k 98.4; out-init s0 10k 23.0, s1 10k 9.7. **The mixer blocks still do not train** (|Wq| 6.510 at 15k, non-zero out init 6.507 at 10k, gate bias 2.000, attention uniform) -> **B37**. Others not finished |
| **B37 step 0 / B43 step 0 (S-118)** | weight-trajectory audit of the 100k runs: only the two mixers are dead (their change equals the weight-decay rate). Model drift vs imagined depth (9 checkpoints, S-121): GTrXL's per-step latent error grows 1.7-3.3x after depth 5 (the training horizon), ResNet's is linear but ~3x higher from the start, policy KL +2-4x in all nine; the 64-sim tree reaches depth 17 |
| B24 / B2 sticky | the 8 evaluators on box 3 (10 B24 checkpoints at 64 sims, 16 sticky evals) had finished nothing when the box was destroyed: rerun (B43 / B38 first) |

Results: `E:\MThesis_EXP\live_20261004_box{1_4x2080Ti_carla, 2_4x2080Ti_carla, 3_4x3090_atari}`, analysis in `E:\MThesis_EXP\analysis_20261004\`. **HF push is pending** (box 3's uplink to HF was 0.2 MB/s,
so the end-of-day save went box -> E: only; push from the laptop). **B36 resume state** (all six runs: `checkpoint_latest.pt` + `replay_latest.npz.gz`, sha256-verified against the box) is under
`live_20261004_box3_4x3090_atari\MThesis\results\100k_benchmark\S058_ezv2_match\S058k_*\checkpoints\`; restore with `gzip -dc replay_latest.npz.gz > replay_latest.npz` and `--resume-from`. s2 is finished; s0 and s1
need ~1 h, s3 ~2.5 h, the out-init runs 3-4 h. **Not saved:** seed 1's env20000 checkpoint (truncated by an interrupted copy, renamed `.CORRUPT_truncated`; its eval and diagnostics JSONs are fine) and out-init seed 1's
env15000 checkpoint. Ops lessons are in S-119 and `scripts/ops/README.md`.
Older status: A14 at 4 seeds, one run each (+7 to +9, CIs touching 0, S-113/S-114); B35 no trunk difference at 30k (explained by the dead mixer).

## New algorithm ideas, 2026-10-04 (S-117, S-118, S-120, S-122, S-123): A36-A58 (CARLA, one front camera) and B37-B46 (Atari)

Evidence ledger (what moved the score, with grades) and the external cross-check: `docs/design/what_has_worked_2026-10-04.md`.

**What the data says** (`E:\MThesis_EXP\analysis_20261004\`, `scripts/analysis/a3_failure_classes.py`; arm J = 4 seeds pooled, ~200 routes of the 220 that E, J
and WoR all drove):
- **Official DS over 220 routes:** E e15 63.4, arm J 58.8-61.9 (seeds), original WoR 49.4. All four J seeds beat WoR paired: +9.8 to +11.9 [CIs exclude 0]. **J is not an overall
  improvement over E: J - E = -2.6 [-5.9, +0.7].**
- **Where we beat WoR:** only on the 102 routes that are neither lead-vehicle nor obstacle scenarios (junctions, signals, turns, pedestrians): J **+22.8 [+16.6, +28.9]**, E +23.8.
  On lead-vehicle scenarios (57 routes: merges, cut-ins, hard-brake, highway exit, actor flows) J is -6.7 [-17.0, +3.5] against WoR, on obstacle scenarios +2.5 [-3.9, +9.5]: ties.
- **J traded one skill for the other:** J - E is **-15.2 [-21.5, -9.0]** on lead-vehicle routes, **+7.2 [+0.7, +13.6]** on obstacle routes, -1.0 elsewhere. Largest family drops J vs E:
  ParkingExit -51, HighwayExit -41, ParkingCutIn -33, MergerIntoSlowTrafficV -24, HardBreakRoute -18 (WoR 100 there). An oracle merge would reach 69.1-69.6 vs E 68.0 and J 64.9.
- **Where the points go (mean lost per run):** J 36.9 = **20.1 vehicle collisions**, 6.7 not finishing, 3.2 layout, 2.4 pedestrian, 2.3 red light, 1.3 stop. WoR 45.0 = 24.1 not
  finishing, 6.5 blocked, 7.4 vehicle collisions. **47% of J runs have a vehicle collision (WoR 22%).** Collisions are 94-98% of the loss on HardBreakRoute and HighwayExit, 89% on
  BlockedIntersection, 87% on MergerIntoSlowTrafficV, 95% pedestrian on DynamicObjectCrossing; ConstructionObstacleTwoWays and AccidentTwoWays lose 46% and 52% to not finishing.
- **Abilities (success %, J seeds / E / WoR):** Overtaking 16-18 / 5.7 / 11.7; Merging 26-39 / 44.9 / 21.8; Emergency brake 25-38 / 32 / 30; Give way 39-50 / 56 / 37; Traffic signs 53-57 / 49 / 44.
- **Headroom and anatomy (S-120; `a3b_headroom.py`, `a3c_collision_anatomy.py`, same records):** (1) J's vehicle-collision points are **80% systematic** (>= 3 of 4 seeds collide on the route; lead 89%, obstacle 83%, other 67%) and 20% lottery.
  (2) **Universal failure = obstacles:** E, J and WoR all score < 50 on 29 of 196 routes (48% of what the best-of-three still loses), 21 of them obstacle families; group means E / J / WoR: lead 82.2 / 66.2 / 71.3, obstacle 44.1 / 51.4 / 48.7,
  other 70.7 / 70.1 / 48.3. (3) **Anatomy:** on obstacle routes E's vehicle collisions are all on the route line (0.70 per run, median lateral offset 0.06 m: it drives straight into the obstacle), J's 83% (0.86); J's layout collisions (0.49 per run,
  cones and warning signs) are half off the line to the left (clipping while passing); J's extra lead-vehicle collisions are on the line too (HardBreakRoute 2.25 vs E 1.25 events per run). (4) **Repeats:** 32% of J's colliding runs have >= 2 vehicle
  events, costing 3.5 points per run (E 1.6, WoR 0.6). (5) **Speed:** J 6.3, E 5.5, WoR 3.5 m/s on finished routes; WoR's safety is bought with the 200 s time cap (24 + 6 + 2 points lost); J's colliding runs are not faster than its clean runs
  on the same route, so the slowdown has to be at the hazard. (6) **Weather:** E is flat across night / rain / fog, WoR loses 13 in fog. (7) **Power:** the smallest difference 220 routes resolve at one run per arm is ~5 DS (7 on the 102
  lead + obstacle routes); J seeds differ by up to 3.3 DS (A52). (8) Camera-only frontier (SimLingo, arXiv v1): Emergency Brake 88 vs our 25-38, Overtaking 57 vs 16-18, Traffic Sign 82 vs 53-57; E's Give Way (56) is at par.
- **What has moved the score (grades in the ledger):** confirmed = route conditioning, the transformer head over WoR's conv head (+15), obstacle data with `route_original`
  (+7.2 on obstacle routes, -15.2 on lead-vehicle routes). **Not established** (CIs include 0, S-079): recovery views, colour augmentation, 288x768. No effect: head size, finer grid,
  loss re-weighting, geometry loss, longer training (`tried_and_ruled_out.md`). The new items are ranked by how much *new information* they inject.
- **Vision is lightly used.** A no-vision MLP on route + speed + command reaches held-out loss 0.7558 vs 0.5793 for the transformer with vision (ch. 13.30), and evals that ran on a
  *random* backbone still scored 50-60 DS (S-072): the painted overlay and the speed token carry most of the score. A0 is open. Hazards are rare in the data, so a head can
  minimise its loss without learning to see them (PlanT 2.0 names the same shortcut problem).
- **The objective is mis-budgeted.** 73.5% of the waypoint loss is the longitudinal coordinate (~ speed x dt); the PID steers from the lateral coordinate and takes speed from the
  target-speed head, whose loss weight is 0.2 and was never swept (ch. 13.26). CarLLaVA / SimLingo fix the same entanglement with a space-indexed path next to the time-indexed
  waypoints: static-layout collisions 0.68 -> 0 and DS 3.21 -> 4.49 (+40%) in CarLLaVA's ablation.
- **LEAD's lens (CVPR 2026, 95.0 DS):** the student is limited by *learner-expert asymmetry*: visibility (the expert reacts to actors the camera cannot see), uncertainty (the expert brakes
  on exact velocities) and intent. Aligning the expert gave +1.37 DS on Bench2Drive and +11 on Longest6 v2, and **its dataset is public** (MIT, 269 GB, all 12 towns, front camera, recovery views,
  per-box `affects_ego`): see A46. Mapping: visibility A21/A38/A42/A45/A46; uncertainty A37/A39/A43; intent A15/A41.
- **Safe2Drive (2026):** SOTA models drop from 95 / 85 DS to ~40 on safety scenarios (late or absent braking for pedestrians, work zones): braking is the field's weak spot too.
- **Atari:** B36 shows the transformer mixer still does not train with weight decay off (|Wq| 6.510 at 15k, init 6.5; non-zero output init: 6.507 at 10k); deeper search at test time lowered
  the score on 4 of 6 checkpoints (B24) while a 64-sim tree reaches depth 17 against a training unroll of 5 (S-116); deterministic evals give ~4 distinct games per checkpoint and
  sticky-action scores fall to ~1/3 (B2).
- **Thread across both domains:** act on what the learned model is *unsure* about (A37/A43 on CARLA, B38 on Atari), use privileged information as labels, never as inputs (A37/A40/A42, B41),
  and audit module-level behaviour before building (A45/A47, B37 step 0): gated residual blocks died from weight decay in both (ch. 13.6c, S-116).

**Tiers.** Tier 0 audits decide what to build and cost almost nothing; Tier 1 is eval-only on checkpoints we already have; Tier 2 are the arms with the strongest external evidence; Tier 3 the rest.

| Tier | ID | Idea | Type / cost | Why (evidence above) | First readout |
|---|---|---|---|---|---|
| 0 | A45 | Vision reliance at hazards: per-frame gain of the head over a no-vision MLP, image ablations (extends A0) | offline, cached features, ~1 box-hour | vision lightly used; collisions dominate | gain on vehicle/walker-hazard frames |
| 0 | A47 | Does the painted route overlay hurt the frozen backbone's view of vehicles? (decides A17c) | offline, 500 frames | overlay sits on the lane where hazards appear | vehicle IoU painted vs clean |
| 0 | A44 | Collision-clip recorder in every eval | 1 day code, free after | 47% of runs collide and we cannot see why (A11 open) | clips on the next eval |
| 0 | B37 | Mixer probe: AdamW param group, open gates, module-update audit | 10-min probe (step 0 done) | B36: blocks at init at 10-15k | |dWq|/|Wq| > 5% by 2k |
| 0 | A52 | Evaluation protocol: mechanism readouts, matched-speed control, >= 3 runs for headline claims | none / one eval config | J seeds differ by up to 3.3 DS; 220 routes resolve ~5 DS, 7 on the 102 lead + obstacle routes (S-120) | every arm table |
| 0 | A58 | More routes: Fail2Drive (200 paired Town13 routes, MIT) next to the 220 | wire the route files; ~2 h per arm | 420 routes resolve ~3.4 DS instead of ~4.9 (S-122) | paired shift gap per arm |
| 0 | A59 | Pooled-token feature cache (exact for the J family): 20 GB memmap, head-only epochs without JPEG decode or encoder | 1 day code, verify in the first 30 min of a box | 17.6 min per epoch on 427k frames; the head pools the map before any learnable layer (S-125) | step time (kill criterion < 3x), loss parity vs the live encoder |
| 0 | A60 | Persistent lane queue for the CARLA harness (route-level jobs, no per-job restart) | 1-1.5 days code + a 40-route parity check | lanes are 47% productive; jobs of 2.7 routes pay 78 s start-up + 43 s hand-over (S-125) | runs/h per box (target ~130), DS parity |
| 1 | A36 | Weight-space merge E <-> J, J-seed soup | eval-only, ~45 min per alpha on 12 lanes | J lost 15 DS on lead-vehicle routes, gained 7 on obstacles | both groups within 3 DS of the better parent |
| 1 | A50 | Contact reflex (stop after the first vehicle collision) + creep clamp | eval-only, small agent change | 32% of J's colliding runs hit again: 3.5 points per run; no expert frame contains a contact (S-120) | events per colliding run, 2nd-event points |
| 1 | A57 | Newsvendor decode: a low speed quantile that relaxes with standing time (+ matched-speed control) | eval-only, ~45 min per config | median decode deadlocked (A28); WoR pays 24 + 6 + 2 points for slowness; asymmetric costs (S-122) | collision events per run at matched speed |
| 1 | B38 | Pessimistic / optimistic search: online-vs-target disagreement, or model-value inconsistency (one network) | eval-only first, 1 box-hour | B24: deeper search hurts; EMCTS / MVI prior art | score non-decreasing in sims |
| 1 | B43 | Depth-capped search (<= training unroll 5) and model error vs imagined depth | eval-only, 1 box-hour | 64-sim trees reach depth 17 > unroll 5; policy KL x2-4 past depth 5 in 9 of 9 checkpoints (S-121) | score vs cap, error curve |
| 2 | B45 | Root-anchored direct multi-step heads as the deep-leaf evaluator | flag + 4 runs, after B43 step 1 | direct multi-step beats iterated forecasting (DLinear study); B43 drift (S-123) | value error and score vs sims |
| 2 | A15 | **Path (space-indexed) + speed head conditioned on the path (aligned cascade), re-budgeted loss, 2.5 s horizon** | one arm, head-only | BevAD 51.7 -> 57.4% SR, static infractions -70%; AlignDrive 89.07 DS; E / J hit stopped vehicles on the line (S-120, S-122) | collision events on obstacle routes, layout collisions, DS |
| 2 | A21+A38 | **Student-aligned labels:** drop/relabel braking caused by actors out of view, anticipatory targets, hazard buckets | 2-3 head-only arms | LEAD +1.37 / +11; CarLLaVA vehicle-hazard buckets; PDM-Lite logs `speed_reduced_by_obj_*` | lead-vehicle DS, completion guard |
| 2 | A37 | Counterfactual speed-safety critic, safety-masked decoding (variant c: candidate paths for the on-line obstacle hits) | one arm, label pass on CPU | 20.1 of 36.9 lost points are vehicle collisions, 80% systematic; E and J hit stopped vehicles on the line (S-120) | AUROC, then lead-vehicle DS |
| 2 | A49 | **Cover the lead-vehicle families in J's fine-tune mix** (arm M) | existing pipeline, ~40 GB download | E never saw them, J's obstacle data overwrote car-following (-15.2) | lead >= E - 3 and obstacle >= J - 3 |
| 2 | A46 | **Train on the LEAD dataset** (state-aligned expert, 12 towns, recovery views), front camera only | adapter + one arm | the field's best recipe published its data | 220-route DS vs E, J |
| 2 | A56 | **SimLingo-Data pilot:** Dreamer counterfactual labels, bucket index, augmented views (same camera as ours) | ~25 GB chunk + labels, one GPU | accelerating crashes in 60.5% of frames of one archive; ready-made A37 / A38 labels (S-122) | critic AUROC by stratum, lead group |
| 2 | A54 | Space-to-depth tokens + 20% token masking | one head-only arm; cache ~4x | BevAD: 36.4 -> 57.4% SR; our 16 tokens are an average pool of a stride-32 map (S-122) | probe AUROC, lead group |
| 2 | A55 | Generative / multi-modal planning head (flow matching or anchor MDN), together with scaled data | head-only; after A56 / A46 data | BevAD: diffusion keeps scaling, point estimators saturate at ~8k scenes (S-122) | overtaking, two-ways families |
| 3 | A39 | Asymmetric ordinal speed loss; speed-head weight 0.2 -> 1 -> 3 | head-only arms | collision x0.6 vs free slowness; weight never swept | collision share |
| 3 | A40 | Clearance (keep-out) loss from logged actor futures | one arm | vehicle + layout collisions = 23 of 36.9 | collision counts |
| 3 | A41 | Two-mode path head (stay / pass), filtered intent gate | one arm | two-ways families 19-23 DS | overtaking success |
| 3 | A42 | Hazard queries on the full-resolution pyramid, box labels | one arm | lead car 0.4-0.8 of a stride-32 cell at brake onset | probes, pedestrian/lead families |
| 3 | A48 | History tokens: previous-frame feature difference (closing speed), keyframe weighting (demoted, S-122) | one arm | CarLLaVA: fewer rear-end collisions but no DS gain; SimLingo uses one image | HardBreakRoute, lead group |
| 3 | A53 | Scale-free waypoint loss, residual to a constant-speed forecast, 2.5 s horizon | one head-only arm | forecasting practice (MASE / RevIN); hazard behaviour at low speed is small in metres (S-122) | skill vs naive at low speed |
| 3 | A43 | Uncertainty-gated caution over the 5 heads (demoted, S-120) | eval-only, ~45 min per config | only 20% of J's collision points are lottery; the heads share one base | lead-vehicle DS, collision share |
| 2 | B37 | Winning mixer setting on the 30k screen, 4 seeds | 4 x 4.5 h | unlocks the thesis test | |Wq|, entropy before scores |
| 3 | B39 | Deeper reanalyze, shallow acting; fresh network for targets | flag + 4 runs | reanalyze sets target quality | 10k score, value error |
| 3 | B40 | Train under sticky actions, report both protocols | 2-4 runs | B2: n = seeds on deterministic evals | 30 distinct games |
| 3 | B41 / B42 / B44 | RAM-supervised auxiliary probe; multi-horizon value heads; stuck-loop / stagnation analysis | diagnostics, flags | see entries | 10k score; episode lengths |
| 3 | B46 | Event / ball-forecast auxiliary (EAWM-style) | one flag | EAWM +10-45% on MBRL baselines; Breakout events are few and exact (S-123) | 10k / 30k score, B43 curves |

**Effort estimate (2026-10-04, S-124; rates from `scripts/analysis/box_rates.py`).** Measured: CARLA eval 65-128 route-runs/h per 12-lane box (plan 90): a 103-route screen 1.1 box-h, a 220-route run 2.2 box-h, +12% reruns; CARLA training 17.6-19.3 min/epoch on 427k frames (live encoder),
a 5-epoch fine-tune ~1.5 h per GPU; Atari 30k steps 3.3-3.5 h per run at one run per GPU (6-8 h at two), 100k ~18-20 h; provisioning 1.0-1.4 h, downloads 11-12 MB/s (34 GB 50 min, 152 GB 3.7 h, 270 GB 6.5 h); a session-day ~5.5 productive box-hours; prices $0.42/h (4x2080 Ti), $0.70/h (4x3090), $0.94/h (2xA40).

| Wave | Items | My implementation | Boxes | Calendar P50 / P90 | Cost |
|---|---|---|---|---|---|
| 1 decisive | B37 probe + 30k screen, A44, A50, A57, A36, A52 control, A58, B43 step 1 + B38 step 0, A56 pilot, A45, A47 | ~7.5 working days | 6 box-days: 3 two-box CARLA-eval days (~2,000 runs), 2 Atari, 1 data box | 6-8 / 10 days | ~$30 |
| 2 top arms | A15 (+A53 horizon), A38, A39, A37 critic + decode, A54, A49 (11 arms) | ~7 days | ~13 box-days: eval ~10 (screens ~2,060 runs, three 3-run 220-route confirmations 1,800 runs), training 3 | 10-14 / 20 days after wave 1 | ~$70-90 |
| 3 the rest | A46 (adapter 4 d, 270 GB, ~11 GPU-h), A55, A40, A41, A42, A48, A43; B45, B46, B39, B40, B41 / B42 / B44, 100k confirmation | ~22 days | ~19 box-days (CARLA 12, Atari 7) | 15-20 / 30 days | ~$100-150 |
| **Total** | 30 items | **~35 working days** | **~38 box-days (~300 box-h)** | **5-6 / 8-9 weeks; waves 1-2 (17 of 30 items): 2.5 / 3.5 weeks** | **~$200-300** (+~$100 if boxes stay overnight) |

Levers: a pooled 4x4-token feature cache (20 GB for J's data instead of 279 GB of full maps) would turn a 1.5 h head-only arm into ~10-25 min if the step is data-bound (the epoch time suggests so; verify in the first 30 minutes), after which evaluation dominates; keep the two CARLA boxes for a whole wave
(re-provisioning costs 1.0-1.4 h per day); drop Tier 3 (A40, A41, A42, A48, A43, B39-B44) to shrink wave 3. Risks: B37 may not train the mixer (+2-4 days), LEAD's camera may not match (A46 -> A56), ~9% crash routes, one box in six lost half a day, 220 routes resolve ~5 DS (claims below ~4 DS need mechanism metrics or A58), Fail2Drive may need extra CARLA assets, the SimLingo licence needs sign-off.

**Compressed plan (2026-10-04 night, S-125: EUR 300 and one month).** The S-124 calendar (36 days) is its implementation time (35 working days): boxes and money are not the limit (EUR 300 is ~$330, ~400 box-hours at ~$0.8/h; the plan used ~300). Measured causes of slow box time (`box_rates.py`, `a3d_screen_design.py`): lanes spend **47% of their time on the route itself**,
29% on per-route overhead (world load, scenario build, agent set-up, teardown; 110 s on Town12/13), 9% on job start-up and 14% on hand-over between jobs of 2.7 routes; a head-only epoch costs 17.6 min because every step decodes 256 JPEGs, paints the overlay and runs the frozen encoder.

| Lever | What | Effect | Status |
|---|---|---|---|
| Parallel implementation | four streams in separate worktrees: **WP1 training** (A59 cache + the head-only flags A15, A53, A38, A39, A37 critic), **WP2 evaluation** (A60 queue, A44, A50, A57, A52 control, A58 routes), **WP3 Atari** (B37, B43 step 1, B38 step 0, B39 / B40 / B42 / B45 / B46 flags), **WP4 data** (A46 adapter, A56 pilot reader, A49 list); I integrate, review and run the boxes (A15 touches WP1 and WP2: model vs agent decode, interface first) | 35 working days -> ~12-14 calendar days | needs your "go parallel" |
| Pooled-token cache (A59) | exact for the J family; 48 KB per frame, 20 GB for 427k frames | head-only epoch 17.6 -> ~2-4 min (projection), a 5-epoch arm ~15 min, 3 trained seeds per finalist ~45 min | verify in the first 30 min of the smoke box |
| Two-stage screen (A52) | stage 1: ~100 lead + obstacle routes, one trained seed, one run (1.1 box-h): resolves 3.0 DS overall-equivalent (6.1 on those routes) vs 4.3 for a full run; stage 2: finalists, 3 trained seeds x 220 routes (6.5 box-h): 3.2 | wave-2 screens 2,060 -> ~1,160 route-runs; ~7,800 route-runs in the whole plan | measured from the records |
| Lane queue (A60) | one persistent evaluator per lane pulling (arm, route) jobs | 417 -> ~320 lane-seconds per run: 104 -> ~134 runs/h (+30%); per-route overhead is the next target once one lane is timestamped | 40-route parity check |
| Wide launches | 6-8 boxes at once (cost follows route-runs, not days); HF reachability first on every box; 1-2 spares (one box in six lost half a day) | a wave in 2-3 days instead of 10-14 | prebuilt balanced queues |
| Provisioning | time HF download vs the CARLA mirror; if HF is >= 3x faster mirror CARLA + maps to the private HF repo; hosts with inet_down >= 500 Mb/s and >= 60 vCPUs (CARLA needs ~5 cores per lane) | 1.0-1.4 h -> ~20 min per box | to measure |

**Schedule** (D1 = 2026-10-05; P50 / P90): D1-D4 build streams (no boxes) -> D4-D5 smoke box (1 box, ~8 h, ~$5: cache build + step time, lane queue parity on 40 routes, one timestamped lane, HF vs mirror speed, dry run of the new flags on 12 routes) -> D5-D8 wide launch A (6 CARLA + 2 Atari boxes: wave 1 and the wave-2 screens, ~$30-55)
-> D9-D13 finalists (3 trained seeds x 220 routes for ~4 arms, merges). In parallel D5-D20: data upgrade (A46 download on a data box from D5, adapter, training on the cache, evaluations) and Atari (B37 probe -> 30k screens -> B45 / B46 / B39-B44 -> the 100k confirmation, 19 h). D20-D26: final E vs WoR vs best on 220 routes x 3 runs + Fail2Drive; buffer to D30.
**Waves 1-2 (17 of 30 items) in ~2 weeks (P50) / 3 (P90); all 30 items in ~4 weeks (P50) / 5.5 (P90); ~13 session-days of yours (S-124: 20).**
**Cost** (measured rates; box prices are earlier listings, today's boxes were not priced): evaluation ~7,800 route-runs = 76 box-h at 104 runs/h (58 at 134) x $0.42-0.94 = $25-70; cached training ~15 GPU-h ~$8; data box ~$10; Atari ~80 GPU-h on 4x3090 ~$20; provisioning, idle tails and saves (14 boxes x 2.7 h) ~$23: **$90-140, with a 25% reserve $115-175 (about EUR 105-160)**,
against the S-124 estimate of $200-300 (+$100 overnight). If time slips, cut in this order: Tier 3 (A40, A41, A42, A43, A48, A53, B39-B44, B46), then A55, then the 100k confirmation. Without parallel implementation the build phase takes ~12 days and the schedule becomes ~5 weeks (P50): drop Tier 3 up front to stay near a month.

**Next sessions.** *CARLA eval-only session (2 cheap boxes, ~4 h):* A44 recorder first; A36 (alpha 0.5 and the J soup on the 103 lead+obstacle routes), A50 (contact reflex + creep clamp), the A52 speed-scale control (J at 0.8 / 0.9 x target speed) and A57 (newsvendor decode) on the same
routes; the same session runs A45 and A47 on one box with the PDM-Lite data (500 frames, cached features). *CARLA training session (data box):* A56 pilot (one SimLingo-Data chunk + its Dreamer labels: critic AUROC; same camera, no adapter) first, then A46 feasibility (one route per scenario family, camera
metadata, route fields) in parallel with A15 and the A21+A38 label arms (one change per arm, head-only on cached features), A37 step 0 (label audit on ~100 archives). *Atari box:* B37 probe (10 min) ->
B36 rerun with the winning setting; B38 step 0 and B43 on B24's 16 checkpoints (eval-only, graph search); resume the saved B36 runs from `E:\MThesis_EXP\live_20261004_box3_4x3090_atari\` only if the
probe says the optimiser is not the fix.

**Order of work for 2026-10-05** (draft; times Athens; ask the destroy time first; end-of-day save must start >= 90 min before it and is sized with a measured uplink) - *if the compressed plan above is approved, items 0-3 become its phases: build streams (no boxes), then the smoke box, then the wide launch*
0. *No box, morning:* push the E: results of 2026-10-04 to HF from the laptop (boxes 1-3 folders, `analysis_20261004`); code: the A44 recorder (agent flag), B37 flags (`--mixer-optimizer adamw --mixer-lr
   --mixer-wd`, GRU gate bias init), `GumbelMCTS(max_depth)` for B43 with a test, the A15 path head; A30 write-up (J vs E vs WoR by scenario group, S-117) and the thesis table. *Box-free work first (S-120; CPU / laptop, E: has 373 GB free; every checkpoint needed is on E:: E e15, J seeds 0-3 e20):* build the A36 merged checkpoints (alpha 0.25 / 0.5 / 0.75, the J soup) and load-test them; A46 step 0 on a LEAD sample; A37 step 0 label calibration (range requests); B43 on the saved B36 checkpoints; A47 / A45 on CPU if the decoders run (500 frames). Rent the CARLA eval and Atari boxes once the A44 recorder, A50 and the B37 flags are coded and unit-tested (a box rented now would only run repeats).
1. *CARLA eval-only session, 2 cheap boxes (`provision_eval.sh` now installs the original WoR; `gen_lanes_0410.py` balances lanes):* A44 recorder on J seed 0 first, then A36 (alpha 0.5 and the J soup on the
   103 lead-vehicle + obstacle routes) and A50 (contact reflex + creep clamp) with the A52 speed-scale control (J at 0.8 / 0.9 x target speed). *One box with the PDM-Lite data (150-500 GB disk):* A45 and A47 audits (500 frames each, ~1 box-hour), A46 feasibility (one LEAD
   route per scenario family, camera metadata, route fields), A37 step 0 label audit on ~100 archives; then the head-only arms A15 and the A21 + A38 label arms (one change per arm).
2. *Atari box, 1 box with 4x 24 GB Ampere/Ada GPUs, >= 64 threads, 150 GB disk; probe HF first and time a 200 MB upload:* B37 probe (10 min per setting) -> the winning setting on the 30k screen (4 seeds) instead of
   resuming the dead-mixer runs unless the probe says the optimiser is not the fix; B38 step 0/0b and B43 steps 0-1 on B24's 16 checkpoints (`restore_b35_checkpoints.py`, graph search); B24 / sticky
   evaluations only as filler (they need ~1.5 CPU cores each and slowed the trainers 1.7x when run 8-wide: keep <= 3 next to training).
3. *Decisions for you:* (i) A46 (the LEAD data) as the main CARLA data upgrade: OK to spend the adapter effort (2-4 days) and ~270 GB of downloads (front camera only if selectable)? (ii) B36: resume the saved runs or
   restart with the fixed optimiser (my advice: wait for the B37 probe); (iii) which CARLA box type (two eval boxes vs one data box) first.
Hardware: CARLA is CPU-bound (~5 cores/lane): high CPU quota; 12 lanes on a 38-CPU quota ran at load 28-36 and slowed WoR routes ~1.8x (530 s vs 290 s in the A9 records; S-119). Atari training needs native bf16 (Ampere or newer); Turing (2080 Ti, Titan RTX) ran CARLA
evals fine on drivers 570/595 but not Atari training. First command on a new box: HF reachability (`getent ahosts huggingface.co`, S-115 box N) and a download/upload speed test. HF token only in `/dev/shm`, never edit
authorized_keys, never `pkill -f` and relaunch in one ssh command. **Saving over a slow link:** gzip the replay buffers on the box (uint8 frames compress ~28x: 817 MB -> 29 MB), hash on the box *after* copying (a trainer that saves
in between makes an early hash stale), and never start a broad overwriting `tar` in the last minutes (an interrupted one truncated a checkpoint).
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

### A14. Obstacle data + the unshifted route (`route_original`) - **Done: +7.2 DS [+0.7, +13.7] over E e15 on obstacle routes (4 seeds, repeat runs), overtaking success ~3x** (S-096, S-097, S-107, S-109, S-111, S-112, S-113, S-115)
2026-10-03 (S-115, final): repeat runs on boxes O + P (60 routes: b2d20 + A9's obstacle list minus the 4 crash routes; E e15 3 runs, each J seed 2 runs). Obstacle-type routes (46), J e20 minus E e15: s0 +7.8 [+0.7, +14.9], s1 +7.0 [+0.5, +14.0], s2 +6.2 [-1.0, +13.6], s3 +7.6 [-0.1, +15.8]; **pooled +7.2 [+0.7, +13.7]**. All 60 routes +4.5 [-1.0, +10.3]; non-obstacle (14) -4.1 [-14.0, +5.1] (n.s., same sign every seed - watch it). Overtaking success (official ability rule) J 15.9-18.3% vs E 5.7%. Still to do: original WoR repeats on the same routes and the 220-route run of arm J (order of work, item 1).
2026-10-03 correction: on all 46 obstacle-type routes (`a9_merge.py --pool`, route XML classification) e20 vs E e15: s0 +8.8 [-0.0, +17.7],
s1 +6.8, s2 +7.5, s3 +7.7; per-route mean of the 4 seeds +6.0 [-1.1, +13.4]. vs the original WoR: E -3.6, arm J +3 to +6, K +7.6
[+0.1, +15.7]. The numbers below (S-113) used the 41-route subset without b2d20's obstacle routes and overstate the effect.
2026-10-02 (S-113): seeds 2 and 3 on box I. vs E e15, obstacle: s2 e18 +8.6, e20 +9.0 [+0.8, +17.7]; s3 e18 +0.4, e20 +11.0
[+3.3, +19.4]. With s1 e20 +9.2 [+1.8, +17.3]: e20 is stable at ~+10 DS, e18 is not. b2d20 unchanged within noise.
2026-10-01 (S-112): arm J seed 1 on box H. Paired vs arm J e18 re-run on the same box: e18 obstacle -11.3 [-21.6, -1.7], e20 -5.1
[-12.2, +1.6]. Paired vs arm E e15 (A9 records): e18 +4.1 [-4.6, +12.4], e20 +9.2 [+1.8, +17.3]; b2d20 within noise. Box effect
(arm J e18 on H vs on P2) only -3.7 on obstacle routes. Report the gain as seed- and epoch-dependent; a third seed decides the size.
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

### A15. Path output next to the speed output - **Next** (promoted 2026-10-04, S-117: the best-evidenced output change; Tier 2)
**Status 2026-10-05 (S-126): model + training code done, smoke-tested on real data.** `train_wor.py --path_head 1 [--path_cascade 1] [--path_loss_weight 1]`: a path head (10 ego-frame points = every second point of the expert's *shifted* route, label from the 20-point `route` at 1 m spacing; zero weights + the nominal straight path as bias, so it starts at 'drive straight'), L1 on the path, optional aligned cascade (`path_embed` zero-initialised and added to the speed head's input: the cascade arm starts exactly as the plain head), new-head resume from E e15 (AdamW state fresh, the 6 new tensors start at their init), loader rebuild from the checkpoint config, `speed_head_loss` / `path_loss` in the epoch metrics. Eval switch `WOR_PATH_STEER=1`: the lateral coordinate of every waypoint is the predicted path interpolated at the waypoint's forward position (logs `[A15] steering from the predicted path`). Smoke (Town01 slice, E e15 resume, cache): path L1 falls 1.4 -> 0.35 over 3 epochs. **Not done:** the steering lookahead of A4 (speed-adaptive d = 0.098 v + 0.192), the 10-waypoint horizon (A53); a closed-loop run needs a trained arm.
TF++, CarLLaVA and SimLingo all predict a space-indexed **path** separately from speed. CarLLaVA: layout
collisions 0.68 -> 0.0; SimLingo: fewer static-object collisions. Target = PDM-Lite's shifted `route` (TF++'s
"path checkpoints", e.g. 10 points at 1 m spacing), lateral PID on the path, longitudinal control from the
existing two-hot target speed. Absorbs the "TF++ path checkpoints" line in `tried_and_ruled_out.md` and
part of A7.
**Re-evidence (S-117).** (1) Our own loss audit (ch. 13.26): 73.5% of the waypoint objective is the longitudinal coordinate, ~ speed x dt, which the head can read from its own speed token;
the PID reads only the lateral coordinate and takes speed from the target-speed head. A space-indexed path removes the coupling: the path loss is purely geometric and the speed head is the one
place speed is decided. (2) CarLLaVA's ablation (Leaderboard 2.0 validation): waypoints only DS 3.21 with 0.68 static collisions; with the path DS 4.49 and 0.0 static collisions (+40%);
SimLingo reports the same effect. J loses 3.2 points per run to layout collisions, and layout collisions are 42-54% of the loss on the construction-obstacle families. (3) **Design:** path
head = lateral offset at 10 fixed distances (1-20 m, TF++'s path checkpoints; target = PDM-Lite's shifted `route`, as above) with L1 plus a smoothness term; waypoints stay for the time side;
steer from the path at a speed-adaptive lookahead (d = 0.098 v + 0.192, A4); loss budget: path weight 1 and target-speed weight 1 instead of 0.2 (A39). One arm on arm J's data, compared with J
on the 103 discriminating routes first, then on all 220.
**Design update (S-122): aligned cascade, horizon, and a correction.** (1) AlignDrive (arXiv 2601.01762) conditions the longitudinal output on the lateral path (speed as a 1-D displacement along the *predicted* path, anchor-based): 89.07 DS / 73.18% SR on Bench2Drive; BevAD (2603.15185) path + speed 57.4% vs waypoints 51.7% SR, static infractions 0.185 -> 0.055. Ours: after the path head, a second decoder stage (cross-attention over the path embeddings and the policy token) predicts the speed bins, so the speed the car chooses is conditioned on the path it will follow; E / J hit stopped vehicles *on the route line* (S-120) because today's speed head has no tie to the path. (2) Horizon: 5 waypoints at 4 Hz is 1.25 s; SimLingo's labels carry 10 (A53). (3) Correction: the "73.5% of the loss" argument above is a share of the L1 *value*, not of the gradient (L1 gradients are sign x weight per element); the case rests on the field's ablations and on S-120's anatomy.

### A16. Oversample obstacle/swerve frames and drop redundant ones - **Done: arm K (`--swerve_frac 0.25`) no gain over arm J** (S-111, S-112)
2026-10-01 (S-112): the natural swerve share in arm J's data is 15.3% (65,522 / 427,310 train frames), so 0.1 was a no-op;
arm K used 0.25 (x1.8). Paired vs arm J e18 (same box): K e18 obstacle -9.6, b2d20 +8.2; K e20 obstacle -0.7 [-9.8, +8.2],
b2d20 -0.7. vs E e15 obstacle: K e20 +13.6 [+6.0, +22.1]. Emphasis on swerve frames adds nothing measurable over J.
`src/training/swerve.py`: a swerve frame = the expert's `route` >0.5 m lateral off `route_original` in the first 4 points (the
`route_shift_stats.py` measure) or one of the `--swerve_window` (8, ~2 s) frames before; the train loader draws them as
`--swerve_frac` of each epoch (seeded WeightedRandomSampler, epoch length unchanged, never below the natural share).
Tests: `tests/test_swerve_sampler.py`. Launch with arm J seed 2 on one box: `scripts/training/launch_armJ2_armK.sh`.
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

### A26. Looming (1/time-to-contact) as the lead-vehicle target - **Later** (with A5/C28; step 3 promoted to A48, S-117)
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

### A28. Read the speed head as a decision, not a mean - **Median tested: -22.9 DS (deadlocks); mean stays** (S-111, S-112)
2026-10-01 (S-112): arm J e18 with `WOR_SPEED_DECODE=median` on the 64 routes arm J ran, paired vs mean decoding: 40.6 vs
63.5, -22.9 [-33.5, -12.6]; Agent got blocked 15 vs 1, TickRuntime 15 vs 7. The median snaps to the stop mode. Only tau > 0.5
or the HMM filter remain; low priority.
`TargetSpeedHead` reads `WOR_SPEED_DECODE` = mean (default, bit-identical to before) | median | quantile:<tau> as the slowest bin
whose CDF reaches tau (not interpolated: interpolation brings the creep back). Eval lane job
`CKD:<label>:<ckpt>:<routes|R19>:<decode>`; the agent log prints `[A28] target-speed decode`. Steps 1-2 done; the HMM filter
(step 3) is not. First run: arm J e18, median, on the 64 routes arm J ran (paired vs j18). Tests: `tests/test_speed_decode.py`.
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

### A9. Full Bench2Drive evaluation (220 routes) - **Done: E vs WoR +14.0 DS; all 5 abilities E 37.3 vs WoR 28.0 (Merging +22 pts, Overtaking -7.5)** (S-093, S-106, S-107, S-108, S-113, S-114, S-115)
Abilities: `py scripts/analysis/a9_abilities.py <same roots> --arms E15,WOR` (official success rule, no CARLA). Traffic_Signs: on the
next CARLA box, dump each route's completion fraction at its first junction waypoint (+8, as the official tool) to JSON and pass `--junctions`.
Re-run any time: `py scripts/analysis/a9_merge.py E:/MThesis_EXP/live_2026092[89]_* E:/MThesis_EXP/live_202610*_* --reference WOR`
(arm J seeds: labels J18/J20 + J{18,20}s{1,2,3}, `--pool J20pool=J20,J20s1,J20s2,J20s3`).
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

### A36. Merge the two skills: E (reactive driving) and J (obstacles) in weight space - **Next** (eval-only; small ceiling, small cost) (S-117)
**Result 2026-10-05 (S-127; lead + obstacle routes, 110 in the set, one run per route, `arm_report.py`):** E15 is not beaten. DS and paired difference against E15 [95% CI]: **M25 67.4, +3.0 [-0.7, +6.9]** (against J +8.1 [+2.8, +13.5]; vehicle collisions per run 0.46 vs E 0.64 vs J 0.86 vs WoR 0.23), M75 63.9 (-0.5 [-5.2, +4.3]), M50 63.1 (-1.8 [-5.6, +1.8]; 41 routes marked infrastructure failure, cause not traced), E+J-soup 61.1 (-3.3 [-7.6, +0.6]), J-soup 60.5 (-4.7 [-9.7, +0.4], level with J). DS falls smoothly from alpha 0.25 to 0.75; everything with much J behaves like J. M25 is the only arm at or above E15 and its gain is not significant (needs a second seed / more routes). The finer alpha arms (0.10 / 0.15 / 0.35 and E+soup 0.15 / 0.25) were built (`E:/MThesis_EXP/prep_20261005/a36/`) but not evaluated: no CARLA box left today.
J is E e15 fine-tuned for 5 epochs on the mixed data (A14), so the two heads sit in one basin and share the frozen backbone: only the ~30M head weights differ.
- **Why:** J - E is -15.2 [-21.5, -9.0] on the 57 lead-vehicle routes and +7.2 [+0.7, +13.6] on the 46 obstacle routes (S-117). Weight interpolation between a base and its
  fine-tune (WiSE-FT, Wortsman 2022) and averaging fine-tunes of one base (model soups) are known to keep most of both skills.
- **Design:** theta(alpha) = (1 - alpha) theta_E + alpha theta_J for alpha in {0.25, 0.5, 0.75}; the J soup = mean of J seeds 0-3; optionally E + J soup. Step 0 (no CARLA): held-out
  validation loss and speed-head cross-entropy along the path, to spot a loss barrier before spending lane hours.
- **Closed loop:** one run per config on the 103 discriminating routes (57 lead-vehicle + 46 obstacle), ~45 min per config on 12 lanes at today's rate (J seeds 2-3 ran
  304 route-runs in ~2.2 h on 12 lanes). Report the two groups separately plus `a9_merge.py` on them.
- **Success:** lead-vehicle group >= E - 3 **and** obstacle group >= J - 3 for the same alpha.
- **Ceiling (honest):** an oracle that uses E on lead-vehicle routes and J on obstacle routes reaches 69.1-69.6 vs E 68.0 and J 64.9 on the 205 routes both drove: +1 to +2 over
  E, +4 over J. The value is a free fix of J's regression and a base for the arms below, not a new capability. If no alpha works, the regression is a data-mix problem: arm M =
  fine-tune from E with an L2-SP penalty toward E's weights and scenario-family-balanced sampling.
- **Single camera:** yes (no input changes).
- **S-120 update:** on the 196 routes all three arms drove, E scores 82.2 on the lead-vehicle families (WoR 71.3, J 66.2) although it never trained on them; J drives 20% faster there (6.96 vs 5.81 m/s) and its extra collisions are on the route line (HardBreakRoute 2.25 vs 1.25 events per run, 8% off the line; ParkingCutIn 1.10 vs 0.40, 0%): J's loss is a braking / speed regression from the obstacle data, not missing coverage. Step 0 should also compare E's and J's speed posteriors on the same lead-vehicle frames (A45's data) and report the oracle as group-level (69.3 on these routes), not per route.
- **Prepared 2026-10-04 (no box):** `scripts/training/a36_merge_checkpoints.py` (refuses to merge unless the frozen backbones are tensor-identical; exact at alpha 0 / 1) built alpha 0.25 / 0.5 / 0.75 (E e15 <-> J seed 0 e20), the J soup (mean of seeds 0-3) and E + soup at 0.5 under
  `E:\MThesis_EXP\prep_2026100536\` (885 MB; each directory = `model_epoch_020.pth` + `frozen_backbone.pth` + `run_config.json` + `MERGE.json`). E's and J's frozen backbones are identical (494 tensors); all five load through `load_wor_model` with 0 missing / 0 unexpected
  keys and finite outputs (CPU forward). On the box: copy the directories to `/workspace/checkpoints/a36/` and use lane jobs `CKL:a36m50_<box>_<port>:a36/alpha0.50_J0/model_epoch_020.pth:<routes>` (labels `a36m25_`, `a36m50_`, `a36m75_`, `a36soup_`, `a36esoup_`; `a9_merge.py` maps them to arms
  M25, M50, M75, MSOUP, MESOUP). Alpha order of work: 0.5 and the soup first on the 103 lead + obstacle routes.

### A37. Counterfactual speed-safety critic (rule distillation) with safety-masked decoding - **Next** (one arm; the main new idea) (S-117)
Imitation sees only the speeds the expert chose, never the unsafe ones, so it cannot learn "this speed would have hit that car". PDM-Lite's logs contain what is needed to compute
it: per-frame 3D boxes with position, extent, yaw, speed and brake state (`aux_heads.py` docstring) and the ego's own future path. Hydra-MDP (NAVSIM 2024) distils rule-based
sub-scores of candidate trajectories into the network the same way: privileged labels, camera-only inference.
**Thesis link (S-117):** World on Rails (Chen et al., ICCV 2021) picks actions by a Q-value learned from rewards that its on-rails world model computes for *counterfactual* actions, which is the
plausible source of its low collision rate (22% of runs, 7.4 lost points); A37 imports that ingredient into an imitation policy offline, for the speed axis, from the logged actor tracks.
- **Labels (CPU, boxes JSON only, no images):** for each frame and each target-speed bin v_j in (0, 4, 8, 10, 13.9, 16, 17.8, 20) m/s, roll the ego along the expert's logged future path
  at constant v_j for 3 s (frames are stored at 4 Hz, `data_save_freq` 5 at 20 Hz), against every logged actor's recorded future footprint (open-loop: actors are assumed not to react
  to the ego). Bin j is unsafe when the footprints come within 0.5 m or the time-to-contact drops below 1.5 s. Pedestrians and cyclists included. The logged
  `speed_reduced_by_obj_id` / `_distance` name the actor that set the expert's speed, so the label pass can start from it; LEAD's data add a per-box `affects_ego` flag (A46).
- **Step 0 audit (before any training, ~100 archives):** the expert's own bin must be labelled safe in >= 98% of frames, otherwise the margins or the open-loop assumption are
  wrong; report the unsafe-label rate per scenario family (expect it high on HardBreakRoute, MergerIntoSlowTraffic, DynamicObjectCrossing).
  **First run on real logs (2026-10-04, `scripts/analysis/pdm_lite_label_audit.py`, HTTP range requests, no download; Town12, 6 routes per archive, 705-913 frames each):** the data support the label
  pass: box ids persist to the next 4 Hz frame in 97-99% of cases, the actor that sets the expert's speed is in the front camera's +-55 degree view in 96-99% of the frames, and the expert's own bin
  (largest bin not above its commanded speed) passes the 1-D inevitable-collision check in 99.3-100% (HardBreakRoute 100%, HighwayExit 100%, MergerIntoSlowTrafficV2 99.4%, DynamicObjectCrossing
  99.3%). **But the label is rarely positive with a lenient criterion** (apply the bin for 0.75 s at <= 2 m/s^2, then brake at 7 m/s^2): fast bins are unsafe in only 4% of in-lane lead frames on
  HardBreakRoute and ~0% on HighwayExit / MergerIntoSlowTrafficV2, because the expert follows at 4-6 m and a 2 m/s^2 acceleration limit hardly changes the speed in 0.75 s. Calibrate the criterion
  (controller-realistic acceleration >= 3.5 m/s^2, hold >= 1 s, or a time-headway rule) until the fastest bins are unsafe in ~20-40% of lead frames while the expert's own bin stays >= 98% safe; a first
  version with a constant-speed (no braking) criterion flagged 44% of the 4 m/s bins unsafe and the expert's own bin 15%: wrong, because it never lets the car brake.
- **Head:** per-bin sigmoid on the speed tokens (BCE, positive weighting for the rare unsafe class), horizons 1 / 2 / 3 s.
- **Decode (an eval-time knob, no retraining):** S = {j : p_safe(j) >= theta}; take the imitation posterior renormalised on S and decode its mean (not the argmax or median, which
  snapped to a mode and deadlocked, A28); if S is empty, the lowest bin. Sweep theta in {0.3, 0.5, 0.7}.
- **Variant (b), reflex:** a small binary head "the expert brakes >= 3 m/s^2 within 1 s" (focal loss), used only as a brake floor when p > theta.
- **Variant (c), candidate paths (S-120):** on obstacle routes E and J drive straight into stopped vehicles (vehicle collisions 100% / 83% on the route line, median lateral offset 0.06 / 0.46 m, 0.70 / 0.86 events per run; WoR 0.22) and J clips cones while passing (layout collisions 0.49 per run, 50% off the line to the left). A speed critic rolled along the *expert's* path calls the straight path at speed safe (the expert swerves), so it has to judge the policy's own path: candidates = {predicted path, +-1.5 m and +-3 m shifts, stop} x speed bins, labels from rolling each against the logged actor footprints (and the static obstacles, if the logged boxes list them: check in step 0), decode = the safest candidate inside the imitation posterior. It is Hydra-MDP's trajectory-vocabulary scoring on one camera and covers the no-reaction and the clipping failures in one mechanism.
- **Ready-made labels (S-122, A56):** SimLingo-Data's `dreamer/` files give, for every frame, counterfactual alternatives (`faster`, `stop`, random `target_speed`, `slower`, `lane_change`, `crash`) with kinematic roll-out verdicts (`dynamic_crash`, `allowed`, `safe_to_execute`); `faster` crashes in 60.5% of the 23,091 frames of one archive, `stop` 46.6%, `target_speed` 43.2%, `slower` 4.3%. Use them instead of (or to calibrate) our own label pass; the images come from the same-camera data chunks (A56).
- **Readouts:** offline AUROC of p_safe per bin, especially in the 2 s before expert hard-brakes; closed loop: lead-vehicle group DS, vehicle-collision share (J 20.1 lost points per
  run; target <= 14), not-finishing share (guard: J 6.7, WoR 24.1; stop if > +3), abilities Emergency_Brake and Merging.
- **Differs from** A28 (decodes the existing posterior), A29 (needs lead-state heads and analytic IDM) and A26 (a looming target): none learns the counterfactual. **Risks:** the
  open-loop assumption is wrong for actors that react to the ego (followers), so label only actors ahead of and beside the ego; a head that always says "unsafe" collapses to WoR's
  stalling, which the guard catches.
- **Cost:** label pass on the data box (CPU), then head-only training on the cached features. **Single camera:** yes; privileged tracks are labels only.

### A38. Anticipatory speed targets, brake-onset sampling and a caution token - **Next** (2-3 head-only arms, label changes only) (S-117)
**Steps 1-2 code done 2026-10-05 (S-126):** `train_wor.py --anticipate_s S` (label = min of the expert's target speed over the next S s; real Town01 frames: mean target 4.06 -> 3.56 / 3.15 / 2.59 m/s at 0.5 / 1 / 2 s, 24-34% of frames lowered) and `--brake_frac F --brake_window W` (brake-onset frames = the 8 frames before a drop >= 4 m/s: 15.3% of Town01 frames; oversampled like arm K's swerve frames, exclusive with `--swerve_frac`); tests. Step 3 (caution token) and the SimLingo bucket index (A56) are not coded.
- **Why:** PDM-Lite brakes on privileged state, sometimes on actors the camera cannot see; LEAD found restricting the expert to camera-visible actors worth +1.37 DS (A21). The
  camera policy sees the same cue later, and the car needs ~0.2-0.3 s to respond. Training on the speed the expert will need soon teaches the policy to slow down for what is about to
  be required. Vehicle collisions are 20.1 of J's 36.9 lost points, and 94-98% of the loss on HardBreakRoute and HighwayExit. **Precedent:** CarLLaVA's training buckets include three
  vehicle-hazard buckets next to acceleration/deceleration, steering, stop-sign, traffic-light, pedestrian and swerve buckets (650k samples per epoch); arm K tested only the swerve bucket.
  The label pass is shared with A21 (visibility-aligned labels). **Real-log facts (2026-10-04):** the expert follows lead vehicles at a bumper gap of 4-6 m (IDM s0 = 4 m, T = 0.25 s); its speed is set by another road user in
  94% (HardBreakRoute) / 81% (HighwayExit) / 41% (MergerIntoSlowTrafficV2) of the frames, in 89-99% of those at a distance under 15 m, while braking (`brake` true) is rare outside HardBreakRoute (34% of frames; 4% HighwayExit, 7% Merger): the speed
  head sees mostly steady following and few brake onsets, which is what the sampler (step 2) rebalances; the caution token (step 3) scales exactly this headway (T and s0).
- **Step 1, anticipation:** label v*(t) = min over tau in [0, Delta] of the expert's target speed at t + tau, Delta in {0.5, 1.0, 2.0} s (2 / 4 / 8 frames at 4 Hz). Waypoints unchanged.
- **Step 2, sampler:** oversample frames in the 2 s before the expert's target speed drops by >= 4 m/s (`--brake_frac`, same machinery as arm K's `--swerve_frac`; K gave no gain on
  swerves, this is the braking side).
- **Ready-made buckets (S-122, A56):** SimLingo's `buckets_paths.pkl` is the published index of such buckets over 2.92M frames: `brake` 1.49M, `leading_object_vehicle` 1.92M, `vehicle_side` 382k, `vehicle_front` 41k, `walker_hazard` 26k, `leading_object_static.prop.trafficwarning` 36k, `changed_route` 177k, `start_from_stop` 97k, `acceleration_-5` 127k.
- **Step 3, caution token (optional):** a scalar c in {0, 1, 2} selects Delta_c in {0, 1, 2} s (or IDM relabelling with the headway T and gap s0 scaled by 1 + c on frames with a lead
  actor); one policy then offers a deployable caution knob that is trained, not decoded, so it does not average modes.
- **Readouts:** lead-vehicle group DS, vehicle-collision share, not-finishing share (stop if > J + 3 points), Emergency_Brake ability; guard: "all other" group >= J - 3.
- **Single camera:** yes. **Cost:** head-only retrain per arm on the feature cache; one change per arm.

### A39. Asymmetric ordinal speed loss (soft caution inside training) - **Next** (cheap, head-only; training-time counterpart of A28) (S-117)
**Status 2026-10-05 (S-126): code done** (`train_wor.py --speed_lambda_over L --speed_lambda_under L`, plain two-hot CE bit-identical at 0; the weight sweep is the existing `--target_speed_loss_weight`); runs on the pooled cache.
Two-hot cross-entropy treats a wrong bin the same whether it is too fast or too slow and how far it is. The metric is asymmetric: a vehicle collision multiplies the route score by
0.6, slowness costs nothing until the 200 s route cap (S-100), and our arms sit at the fast end (47% of runs collide) while WoR sits at the slow end.
- **Loss:** CE_twohot + lambda_o * sum_j p_j * relu(v_j - v*)^2 / v_max^2 + lambda_u * sum_j p_j * relu(v* - v_j)^2 / v_max^2, with lambda_o = 2-4 x lambda_u; optional HL-Gauss
  smoothed targets (B9's trick). Decode stays the mean, so there is no mode snapping and no deadlock like the median decode (-22.9 DS, A28) or the hard brake (S-059).
- **Arms:** (lambda_o, lambda_u) in {(0.5, 0.25), (1, 0.25), (2, 0.5)}; one change per arm; compare with A38 in the same table. **Also the plain weight sweep that was never run:**
  `target_speed_loss_weight` in {0.2 (today), 1, 3} with the unchanged two-hot CE: the head decides longitudinal control and carries 0.2 of an objective dominated by the longitudinal waypoint
  axis (ch. 13.26); no sweep of this weight is recorded in the logs.
- **Readouts:** vehicle-collision share down without the not-finishing share rising by more than 5 points; lead-vehicle group DS. **Kill:** the posterior simply shifts slow everywhere.
- **Single camera:** yes. **Cost:** head-only.

### A40. Clearance (keep-out) loss from logged actor futures - **Next** (one arm; shares A37's tracks) (S-117)
Waypoints are trained with L1 against the expert path only. Add a hinge on the predicted waypoints: for waypoint k at time t_k, max(0, r - d_k)^2 where d_k is the distance to the
nearest logged actor footprint at t + t_k (transformed into the ego frame of t, actor tracks stop-gradient), r = ego half-width + 0.5 m. The expert path itself is collision-free in
the logs, so the loss is zero on expert-like predictions and only pushes back when the network drifts toward another road user (an inequality constraint, not a second target).
A longitudinal version penalises the waypoint spacing (implied speed) that overruns the gap to the nearest in-corridor actor.
- **Why:** vehicle plus layout collisions are 23.3 of J's 36.9 lost points; layout collisions alone are 42-54% of the loss on the construction-obstacle families. S-120: J's layout collisions on obstacle routes (0.49 per run; cones 95, warning signs 64) are half off the route line, to the left: it starts the pass and clips the cones.
- **Design notes:** use only vehicles and pedestrians ahead of or beside the ego (corridor 3 m wide); start with weight 0.1 of the waypoint loss; log the share of frames where the
  hinge is active. Risk: actors that reacted to the expert's own motion make the recorded future a counterfactual for the learner (passing manoeuvres), hence the corridor limit.
- **Readouts:** `collisions_vehicle` and `collisions_layout` events per route, ADE/lateral error unchanged (guard +0.01 m), lead-vehicle and obstacle groups.
- **Single camera:** yes. **Cost:** one arm; the track extraction is the A37 label pass.

### A41. Two-mode path head (stay / pass) gated by a filtered intent posterior - **Next** (one arm) (S-117)
The two-ways obstacle families are J's worst (ConstructionObstacleTwoWays 19.5, AccidentTwoWays 22.6, ParkedObstacleTwoWays 47.3) and the first two lose 46% and 52% of their points to the car
stopping and never passing; overtaking success is only 16-18%. Passing is a committed multi-second manoeuvre, and a single-frame regression head can flip between "stay" and "pass" from frame to frame.
- **Labels:** pass = PDM-Lite's `changed_route` flag, or the lateral offset between the shifted `route` and `route_original` above 0.5 m within the next 3 s (the pair A14 already reads).
- **Design:** two path/speed heads (stay, pass) trained on the frames of their mode, and an intent classifier (BCE) as the gate. At inference, filter the intent posterior with a sticky
  two-state HMM (enter at 0.7, leave at 0.3, minimum dwell 2 s) and output the selected head. It is A23's scenario experts with a binary label and a temporal filter, so it adds no
  history to the network and no copycat shortcut (Wen et al.).
- **Readouts:** overtaking success (J 16-18%), the three two-ways families, not-finishing share on obstacle routes; guard: layout/vehicle collisions on obstacle routes must not rise
  (a wrong-mode commitment into oncoming traffic is the failure to watch) and the lead-vehicle group >= J - 3.
- **Single camera:** yes. **Cost:** one arm.

### A42. Lead-hazard state heads on the full-resolution pyramid, supervised with box labels - **Next** (one arm; extends A5, A31, A34) (S-117)
A5/A34 note that an auxiliary loss on the *existing* 4x4 tokens cannot add information the frozen backbone did not already expose to the head. The head however sees only a 4x4
pooling of the stride-32 map: at the brake-onset gap a lead car covers 0.4-0.8 of one stride-32 cell (S-100 section 8.4), a brake light or a child is below that. `vision_grid 8` did
not help (ch. 13.24) but it is a finer grid of the same stride-32 map, so it does not test this.
- **Design:** K = 6 learned queries cross-attend (2D sin-cos positions) over the stride-16 + stride-32 maps (`CarlaPretrainedEncoder.forward_pyramid`, ~1,080 tokens at 288x768; the
  stride-8 map only in a corridor window as a second step). The K query outputs are appended to the policy's token set.
- **Supervision (loss only, from the logged boxes):** query 0 takes the actor named by `speed_reduced_by_obj_id` (the one that set the expert's speed); query k is matched to the k-th nearest
  in-corridor actor by along-track distance: presence (focal BCE), class (vehicle /
  pedestrian / cyclist), log distance, closing speed, brake-light state (the box's `brake` flag), lateral offset. Weights 0.05-0.2 of the waypoint loss.
- **Controls:** (i) the same queries with the auxiliary loss off (separates "more tokens" from "supervision"); (ii) loss on the 4x4 tokens only.
- **Readouts:** offline probes (distance error at 20-30 m, brake-light AUROC); closed loop: lead-vehicle, pedestrian families (VehicleTurningRoutePedestrian 22.1, DynamicObjectCrossing 59.0).
- **Single camera:** yes; boxes are labels. **Cost:** one arm; the cache must hold the stride-16 features (about 4x the current cache): check disk first.

### A43. Uncertainty-gated caution with a stall breaker, over the heads we already have - **Later** (demoted from Next, S-120; eval-only; extends A33) (S-117)
**Demoted to Tier 3 (S-120).** Of J's vehicle-collision points 80% sit on routes where >= 3 of the 4 J seeds collide (lead 89%, obstacle 83%, other 67%) and only 20% where 1-2 do. The five heads share one base and agree on the systematic cases, so the disagreement signal can reach the 20% (4 of 19.7 points) plus part of the 3/4 class (an upper bound of ~9 points), below what 220 routes resolve (A52); the plain 5-head mean (A33) is the cheaper first step. Its stall breaker and the creep clamp moved to A50.
We hold five heads on the same frozen backbone (E and J seeds 0-3), so one backbone pass plus five head passes gives an epistemic signal today. Run-to-run noise on our arms is a
collision lottery (S-101), and 47% of runs collide.
- **Rule:** u_t = standard deviation across heads of the expected speed (or mean pairwise Jensen-Shannon distance of the speed posteriors); target speed = mean - kappa * u_t (kappa ~1.5),
  floored at 0. **Stall breaker:** if the ego has been below 0.5 m/s for > 4 s and the heads' mean p(stop) < 0.9, use the ungated speed for 3 s (at most twice a minute). The earlier hard
  rule (P(stop) > 0.9 -> target 0) deadlocked three routes because the target stayed pinned at 0 with full brake and the creep overridden (S-059); here the caution is soft and released.
- **Configs on the 103 discriminating routes (~45 min each):** single J seed; plain 5-head mean (A33); gated; gated + 2 photometric test-time views.
- **Readouts:** lead-vehicle group DS, vehicle-collision share, not-finishing share, sigma_run on the lottery routes (23687, 2143, 3717, 3936); also whether u_t predicts collisions
  (needs A44's clips).
- **Single camera:** yes (test-time views are of the same frame). **Cost:** eval-only.

### A44. Collision-clip recorder in every eval, then failure-conditioned fine-tuning - **Next** (infrastructure first; enables A37, A38, A40) (S-117)
**Status 2026-10-05 (S-126): code done** (`src/eval/clip_recorder.py` + agent glue behind `B2D_CLIPS_DIR`, wrapped so a recorder error cannot cost a route; unit-tested without CARLA, **not yet run in CARLA**: first closed-loop use = the smoke box). Use it through `CKE` lane jobs.
J has a vehicle collision in 47% of its runs, and we still cannot say who hit whom at what speed or what the speed head predicted (A11 is open, the evals store only JSON).
- **Recorder (`bench2drive_agent.py`, flag `B2D_CLIPS_DIR`):** a ring buffer of the last 6 s at 4 Hz: the downsized frame (JPEG), waypoints, speed posterior, target speed, ego speed and
  control, and for analysis only the other actors' poses from CARLA (never given to the policy). Written when a vehicle / pedestrian / layout collision fires, and on
  `vehicle_blocked` and route timeout. About 1 MB per clip, ~400 clips for 220 routes x 4 seeds.
- **Use 1, taxonomy (finishes A3/A11):** rear-end vs side vs crossing, ego speed at impact, did the speed head ask for a stop, how far was the other actor when the posterior moved.
- **Use 2, training signal without an expert label:** unlikelihood on the speed bins >= the executed speed for frames in [-2 s, -0.5 s] before a rear-end or pedestrian collision
  (avoidable by slowing). Needs balanced positives from similar situations we passed, or it teaches blanket caution; do use 1 first.
- **Use 3:** relabel those states with PDM-Lite replay (A6's shadow-mode recipe) once the clips say where it matters.
- **Use 4 (precedent, S-122):** VLAAD (arXiv 2603.25946) learns a collision-risk score from 1,521 collision clips of TF++ failures (Town12 / 13, 4 Hz RGB + control; no release found) and appends it to TF++'s global state: Town13 DS +14.1% relative, Bench2Drive 86.97 DS / 71.97% SR. Our clips are on-policy, so the same recipe on J's failures is use 2 / 3.
- **Single camera:** yes; actor poses are logged for analysis only. **Cost:** about a day of code, free per eval afterwards.

### A45. Vision reliance at hazards: does the head use the image where it matters? - **Next** (Tier 0 audit; offline on cached features, ~1 box-hour; extends A0) (S-117)
**Evidence:** a no-vision MLP on route + speed + command reaches held-out loss 0.7558 against 0.5793 for the transformer with vision (ch. 13.30: lateral error 0.0713 -> 0.0503 m);
evals that accidentally ran on a random backbone still scored 50-60 DS (S-072); 47% of J's runs have a vehicle collision. Hazards are rare in the data, so a head can drive the loss down
without learning to see them (PlanT 2.0, 2025, names exploitable shortcuts and expert rigidity as structural flaws).
- **Strata (verified on real logs 2026-10-04):** of the 856 speed-reduction frames of DynamicObjectCrossing the cause is a vehicle in 74%, a traffic light or sign in 19% and a pedestrian in 7%; PDM-Lite logs per frame the reason for the expert's speed (`speed_reduced_by_obj_type`, `_id`, `_distance`; `vehicle_affecting_id`, `walker_affecting_id`; `brake`). Stratify
  held-out frames by hazard type (vehicle / walker / none) and distance (< 15, 15-30, 30-50 m).
- **Measure per stratum:** target-speed cross-entropy and expected-speed error for (a) the full model, (b) image tokens zeroed, (c) image tokens from another frame of the same route,
  (d) a random backbone, (e) the no-vision MLP (`scripts/analysis/audit_route_leakage.py` already fits it). **Vision gain** = loss(e) - loss(a). Plus a linear probe on the frozen 4x4 tokens
  for "vehicle within 25 m and closing" (does the *feature* carry the cue?).
- **Reading:** gain ~ 0 on hazard frames but probe AUROC high -> the head ignores a cue it could read: forcing signals are the right class (A37, A38, A39, A42 loss, A21). Probe AUROC low
  -> access problem: A42 / A31 / A34 / A47. Gain large but the closed loop still collides -> timing and decision (A38, A43). Closed-loop complement: A0 (blank image) on the 103
  lead + obstacle routes.
- **Single camera:** yes. **Cost:** ~1 box-hour on the cached features; no CARLA.

### A46. Train on the LEAD dataset (state-aligned expert, all towns, recovery views), front camera only - **Next** (data upgrade; feasibility first) (S-117)
LEAD (CVPR 2026, TransFuser v6) is the best open recipe (95.0 DS; camera-only 360 degrees 91.6). Its largest single ingredient is a *state-aligned expert*: it ignores actors outside the camera
view, limits traffic-light reasoning to the frustum, caps speed by observable flow, brakes earlier near visible hazards, enlarges boxes at turns (+1.37 DS on Bench2Drive, +11 on Longest6 v2).
**The data are public:** HF `ln2697/lead` (MIT, 269 GB, 45 scenario folders; 8,930 routes, 43 scenario types, all 12 CARLA towns), six RGB cameras at 4 Hz, ego state / boxes / route /
traffic lights at 20 Hz in Arrow (`py123d` format), per-box `affects_ego` and occlusion counts, perturbed ("recovery") views (shift 0.1-1.0 m, yaw 5-12.5 degrees, the same idea as our
recovery camera), camera-only checkpoints (`ln2697/transfuser-carla-123d`). Our PDM-Lite set has six towns and no aligned expert (69% of Bench2Drive is Town12/13).
- **Step 0, feasibility (1 day, one box):** download one route per scenario family (their `scripts/download_one_route.sh`); identify the camera closest to our front camera and read its
  intrinsics (`scene.get_camera_metadatas()`); mount an identical camera in our eval agent; read the route / target-point fields to build `route_original`; check how target speed and
  waypoints are stored (our head predicts time-indexed waypoints plus a two-hot target speed over TF++'s 8 bins); estimate the front-camera-only size (the files are per camera).
- **Step 1, adapter (2-4 days):** `py123d` Arrow -> our loader (frame, speed, route, waypoints from future ego poses, target-speed label, boxes).
- **Step 2, arm L:** arm E's recipe (frozen TF++ regnety_032, route overlay, transformer head) on the LEAD front-camera data; compare with E and J on the 220 routes (all four ability
  groups); a matched WoR-head arm on the same data keeps the fair-baseline rule (retrain allowed, no tuning).
- **Step 3 (optional, A7):** the public camera-only TFv6 checkpoint as a teacher for soft speed / path labels (its calibrated uncertainty is exactly what the aligned expert encodes).
- **Risks:** adapter effort and convention mismatches (camera intrinsics, speed bins, 4 Hz vs 20 Hz); the 220 evaluation routes may overlap LEAD's training scenarios (report as a caveat); it
  makes the gap to WoR larger partly through data, so report it as a data-upgrade row, not as a method row.
- **Camera compatibility first (S-122):** our frozen TF++ encoder was trained on 1024x512, FOV 110, camera at x = -1.5 m, z = 2.0 m. The paper ablates a 110 degree front camera, but the README does not state the released rig's per-camera calibration: check it in step 0. Bench2Drive's data camera (1600x900, FOV 70, x = 0.8, z = 1.6) and TaCarla's (nuScenes-style) differ and cannot feed the frozen backbone without re-rendering; **SimLingo-Data uses our camera (A56) and can start first**.
- **Single camera:** yes (one of the six cameras; the others are never read). **Why Tier 2 first:** data alignment was the largest lever in the field's best recipe, and our own J/E trade-off shows
  the data mix sets the skills.

### A47. Does the painted route overlay hurt the frozen backbone's view of vehicles? - **Next** (Tier 0 audit, offline, 500 frames; decides A17c) (S-117)
The route polyline (alpha 0.55, 5 px) is painted into the RGB before the frozen TF++ backbone reads it, over exactly the lane where lead vehicles and crossing pedestrians appear; TF++ was
trained on clean images. S-072 also shows the route survives a *random* backbone, i.e. the overlay carries its information through the image.
- **Test:** run the TF++ decoders (A34 step 1, `scripts/analysis/tfpp_decoders_rgb_only.py`) on the same 500 frames with and without the overlay; report vehicle and pedestrian IoU, depth error in
  the route corridor, and the corridor-only IoU of the frozen features' semantic head.
- **Reading:** vehicle IoU in the corridor drops by > 2 points -> arm: the route enters as a separate token / channel and the image stays clean (A17c; keep both the route tokens and a *post-backbone*
  route map). No drop -> close the question.
- **Single camera:** yes. **Cost:** ~1 box-hour, no training.

### A48. History tokens: previous-frame feature difference as a closing-speed cue, with keyframe weighting - **Next** (one arm; promotes A26 step 3) (S-117)
One frame cannot give relative speed, and the lead-vehicle families (HardBreakRoute, MergerIntoSlowTraffic, HighwayExit: 87-98% collisions) are exactly where closing speed decides. CarLLaVA reports
fewer rear-end collisions with temporal input (qualitative; its leaderboard score did not rise) and names rear-end collisions and high-speed merging as its failure modes, the same as ours.
- **S-120 evidence:** J's extra lead-vehicle collisions are on the route line (HardBreakRoute 2.25 vs E 1.25 events per run, 8% off the line; ParkingCutIn 1.10 vs 0.40, 0%) and J drives 20% faster there: a closing-speed / braking failure, which is what history tokens, A38, A39 and A42 address; HighwayExit (30% off the line) and ParkingExit (80%) are lateral.
- **Demoted (S-122):** the strongest single-camera system on the leaderboard (SimLingo, 86.55 DS) processes one image (two 448x448 tiles) without frame stacking, and CarLLaVA saw no DS gain from temporal input; run this only if the lead group still regresses after A56 / A54 / A15.
- **Design:** extra tokens = pooled 4x4 features of frame t minus frame t-k (k = 1 or 2 saved frames = 0.25 / 0.5 s; the agent buffers 5 / 10 ticks), **no past actions** (no copycat shortcut),
  keyframe up-weighting at expert action changes (Wen et al., ICML 2021, shown in CARLA). Cached features make the second frame nearly free in training.
- **Readouts:** lead-vehicle group DS and vehicle-collision share, HardBreakRoute; guard: the non-lead groups >= J - 3. **Kill:** no gain on the lead group.
- **Single camera:** yes (same camera, two time steps). **Cost:** one arm; compare with A42 (spatial detail) since they attack the same collisions from two sides.

### A49. Cover the lead-vehicle scenario families in the fine-tune mix (arm M) - **Next** (data-mix repair of J with the existing pipeline; ~40 GB per town; Tier 2) (S-117)
J's fine-tune added only the 10 obstacle archives per town (Town12/13) and lost 15.2 DS on lead-vehicle routes against E (S-117). **E never saw those families either**: the 6-town set holds only ControlLoss, DynamicObjectCrossing,
OppositeVehicleRunningRedLight, SignalizedJunctionLeft/RightTurn, VehicleTurningRoute and NoScenario (S-062); its skill there is generic car-following, which J's obstacle data partly overwrote (a car-following policy that has learned
"swerve around what is ahead" collides where the lead vehicle is slow, not blocking; S-097 warned of false overtakes). The matching archives exist and are small: Town12 `HardBreakRoute` 3.2 GB, `HighwayCutIn` 3.3, `HighwayExit` 3.8,
`MergerIntoSlowTraffic` 2.5 + `V2` 4.2, `InterurbanActorFlow` 2.4 + `AdvancedActorFlow` 1.9, `StaticCutIn` 2.2, `ParkingCutIn` 4.5, `ParkingExit` 4.0, `CrossingBicycleFlow` 2.4, `EnterActorFlow` 1.7 + `V2` 2.0,
`BlockedIntersection` 2.5: ~41 GB (HF file listing; check Town13). S-063's "coverage alone did not fix the obstacles" was a route-input mismatch (S-096), which does not exist for families without route shifts.
- **Arm M:** J's recipe (E e15 fine-tuned 5 epochs on E's schedule, `--route_key route_original`) on the J mix plus these archives, mixed by frames (base : obstacle : lead-vehicle about 2 : 1 : 1); one change versus J.
- **Readouts:** the 103 discriminating routes first, then all 220; **success:** lead-vehicle group >= E - 3 and obstacle group >= J - 3 at the same time (what A36 tries in weight space); also the share of J's vehicle collisions that
  remain on the lead families.
- **Relation to the others:** the cheap data-side twin of A36 (no new code) and the small version of A46 (LEAD's data cover every family with an aligned expert but need an adapter); run A36 first (eval-only), A49 if the merge has no
  good alpha.
- **Single camera:** yes. **Cost:** ~40-80 GB download (Town12 +/- Town13), 5 epochs of fine-tuning, J's evaluation protocol.

### A50. Contact reflex (stop after the first vehicle collision) and creep clamp - **Next** (eval-only; small agent change; Tier 1) (S-120)
**Status 2026-10-05 (S-126): code done** (`src/agents/safety_layer.py`: `WOR_CONTACT_REFLEX=1`, `WOR_CREEP_CLAMP=1`, 9 unit tests incl. end-to-end through `QwenWorldOnRailsPolicy.act`); not yet run in CARLA. Lane job: `CKE:<label>:<ckpt>:<routes>:WOR_CONTACT_REFLEX=1,WOR_CREEP_CLAMP=1`; the log shows `[A50/A57] safety layer: ...` (grep it to verify a lane really ran it).
J's runs with a vehicle collision have >= 2 vehicle events in 32% (E 17%, WoR 17%), and the 2nd and later events cost J 3.5 points per route-run (lead-vehicle routes 5.7, obstacle 4.3; E 1.6, WoR 0.6; S-120). Each registered collision multiplies the score by 0.6; the
leaderboard ignores a contact at ego speed < 0.1 m/s and a repeat with the same actor within 5 s or 5 m, so a stopped car cannot be penalised again. No expert frame contains a contact, so the policy has no learned reaction to one and drives on (median 9.4 m between
consecutive events; S-101: ~5 m = creeping into a parked car, which is what the mean decode of a stop / go posterior produces: P(stop) = 0.5 -> a 4 m/s creep, A28).
- **Reflex (speedometer only, an input we already use):** a contact signature = speed drop >= 2.5 m/s within 5 ticks (0.25 s) while the commanded brake was < 0.5 -> hold the brake for 3 s, then release through the stall breaker of A43 (move again only if the
  posterior's stop mass is < 0.5). Tune the thresholds on A44's clips.
- **Creep clamp:** when the ego speed is < 3 m/s and the speed posterior has stop mass >= 0.3, command 0 instead of the mean until the stall breaker (standing > 4 s, stop mass < 0.9) releases it, so a bimodal "stop or go" posterior cannot average into a creep toward the obstacle.
- **Readouts (mechanism metrics, A52):** events per colliding run (J 1.5 -> <= 1.15), points lost to the 2nd+ event (3.5 -> <= 1), vehicle collisions on the obstacle and lead groups, time-cap share (guard: <= +3 points, J 5.0). J seed 0 against itself on the 103
  discriminating routes. The DS upper bound is 3.5 points, inside the noise of a DS comparison: judge it by the mechanism metrics.
- **Honest limit:** a safety layer, not a driving skill; report it as its own row. **Single camera:** yes (speedometer and the policy's own heads). **Cost:** a day at most, eval-only.

### A52. Evaluation protocol: what 220 routes resolve, mechanism readouts, and a matched-speed control - **Next** (Tier 0; analysis plus one eval config) (S-120)
**Why:** two J seeds (same recipe) differ by up to 3.3 DS in the mean over 201 routes (the six pair differences spread +-1.9); the smallest paired difference with 80% power is 4.9 DS at one run per arm (5.7 with the route x arm interaction of J - E) and 6.9 on the 102 lead +
obstacle routes (8.3), 3.9 / 5.7 with four runs per arm; the floor is ~3.1 / 4.6 however many runs, because routes carry the interaction. 3 DS needs ~540 routes at one run per arm, 2 DS ~1,200. Closed-loop DS cannot resolve 2-3 point effects at any run count on 220 routes,
which is why the one-seed arms of the last weeks ended with CIs touching 0.
- **Predict, then run:** only arms whose target-group effect is expected to be >= 8 DS (lead 52-57 routes, obstacle 45-46) get a DS verdict; the others are judged by **mechanism metrics** that have far more power: vehicle-collision events per run by group, share of runs with a
  collision, events per colliding run (A50), off-line share, time-cap share, mean speed, overtaking success. Report DS next to them, not instead of them.
- **Headline claims** (thesis table): >= 3 runs per arm on all 220 routes (SE of the difference ~2).
- **Matched-speed control for every caution method (A37, A38, A39, A43, A50):** a method that lowers collisions by driving slower has learned nothing. Compare with the plain head at a global speed scale (0.8 / 0.9 / 1.0 x target speed) at the same mean speed. WoR is safe by driving
  1.8 m/s on average (3.5 on finished routes) and loses 24 + 6 + 2 points to the time cap (S-120); an open-loop vs closed-loop study (arXiv 2605.00066, NAVSIM vs Bench2Drive) finds the same pattern: methods that buy safety with progress rank high open-loop and fall in closed loop
  through timeouts, and ego progress is the strongest single predictor of closed-loop success.
- **If a 2-3 point effect must be resolved** (a thesis ablation), add routes, not runs; first candidate: **Fail2Drive's 200 paired Town13 routes (A58; MIT; 420 routes resolve ~3.4 DS at one run per arm)**, then further scenario instances from the Bench2Drive route generator.
- **Screen protocol (S-125, `a3d_screen_design.py`):** the noise of one route-run is 74% evaluation-only (repeats of one checkpoint on the 46 obstacle-heavy routes that carry them: J variance 219, SD 14.8; E 65; WoR 116) and 26% training seed (J 76, SD 8.7): repeat runs help, but a trained seed's own deviation stays. **Stage 1** (every arm): one trained seed, one run, on the ~100 lead + obstacle routes
  (49% of the routes, 57% of J's lost points, 49% of the noise; 1.1 box-h): resolves 3.0 overall-DS equivalent (6.1 on those routes; arm x route interaction SD 10) against 4.3 for a 196-route run; it sees only effects that sit on those routes. **Stage 2** (finalists): 3 trained seeds on all 220 routes (6.5 box-h): 3.2. Only 19 of 196 routes are clean (100 in every run of every arm): they are the finalists' regression check, not a saving.
- **Single camera:** n/a. **Cost:** none for the policy; one extra eval config for the speed-scale control (~45 min on 12 lanes).

### A53. Scale-free waypoint loss, residual to a constant-speed forecast, and a 2.5 s horizon - **Next** (one head-only arm; time-series practice; Tier 3) (S-122)
Today: 5 waypoints at 4 Hz (a 1.25 s horizon), L1 in metres, lateral weight 3.0. Forecasters are trained and scored on scale-free errors (MASE: error divided by a naive forecast's error; RevIN: instance normalisation) so that large-scale series do not dominate,
and a model is judged by its skill over the naive forecast (constant velocity is the standard strong baseline for vehicles). A metre-scale L1 weights fast frames more (the error scale is speed x dt) while hazard behaviour (stopping, starting, creeping) lives at low
speed. Note: L1 gradients are sign x weight per element, so the 73.5% "longitudinal share" of ch. 13.26 is a share of the loss value, not of the gradient (S-122); the argument here is scale across frames.
- **Design:** (i) target = real waypoint minus the constant-speed, route-following forecast (v dt k along the route points); the network predicts the residual (zero-initialised head); (ii) divide each waypoint's error by max(naive displacement, 1 m) or by the
  per-horizon MAE of the naive forecast on the training set; (iii) horizon 5 -> 10 waypoints (2.5 s, as SimLingo's labels), the PID keeps reading the first 2-3; (iv) readouts: skill score against the naive forecast per stratum (speed < 3 m/s, lead vehicle within 15 m,
  brake onset), speed-bin cross-entropy on hazard frames; closed loop only if the low-speed / hazard skill improves by >= 20%. Expect a small effect: judge it by mechanism metrics (A52).
- **Single camera:** yes. **Cost:** head-only on cached features; combine with A15 (the same arm can carry the longer horizon).

### A54. Space-to-depth tokens with token masking (BevAD) - **Next** (one head-only arm; Tier 2) (S-122)
**Status 2026-10-05 (S-126): code done, smoke-tested on real data (live encoder).** `train_wor.py --token_mode s2d --s2d_patch 3 --token_mask 0.2 --vision_grid 6x16` (288x768: stride-16 map 18x48 -> pixel-unshuffle p = 3 -> 6x16 = 96 tokens of 576 x 9 channels; lateral column masking in training only). **Cannot use the pooled cache** (it holds the stride-32 map; the arm trains with the live encoder, ~17-40 min per epoch) and **cannot resume the vision input from E e15**: `vision_proj` and `trunk.vision_pos` have another shape, they start from their own init, the rest of E's trunk is kept and AdamW starts fresh (`load_trainable_state`: only those two names may be reshaped). So the arm is a fair test only with more epochs than J's 5 (the new vision projection has to be learned): plan 15 epochs (~5 h on one 3090) and compare to J and the cached controls.
**Evidence:** BevAD (arXiv 2603.15185; six cameras, Bench2Drive), Table 1: baseline 36.4% SR / 66.9 DS; masking 20% of the lateral cells 42.0 / 72.4; pixel-unshuffle (space-to-depth) with patch 4 **57.4 / 82.6**; patch 5 collapses (40.9 / 66.4); the unmasked model attends to distant,
occluded or irrelevant cells (causal confusion), and the gains are invisible in open-loop L1. **Ours:** the stride-32 map (9x24 at 288x768) is average-pooled to 4x4 = 16 tokens; a lead car at brake onset is 0.4-0.55 of one stride-32 cell (S-100) and is averaged with its
surroundings; `vision_grid 8` pooled the same map (ch. 13.24); vision is lightly used (A45).
- **Design:** stride-16 map (18x48) -> pixel-unshuffle p = 3 -> 6x16 = 96 tokens of 9x channels -> linear to d (nothing averaged away); random masking of 20% of the lateral columns in training only; arms: unshuffle + masking, unshuffle only, the 16-token baseline; the cache holds the
  stride-16 features (~4x the current cache: check disk).
- **Readouts:** A45's probe AUROC on the new tokens; lead-vehicle group DS and collision events per run; guard: other groups >= J - 3. **Single camera:** yes. **Cost:** one arm; relates to A42 (queries over the pyramid) and A31.

### A55. Generative / multi-modal planning head, together with scaled data - **Later** (head-only; Tier 2, conditional on A56 / A46 data) (S-122)
**Evidence:** BevAD (Table 2 and the data-scaling figure): point estimator + waypoints 51.7% SR, diffusion 56.2%; point + path / speed 57.4%, diffusion + path / speed 59.4%; dynamic infractions 0.423 (diffusion) vs 0.505; with cumulative data splits to ~16k scenes diffusion
improves **linearly** while point estimators show diminishing returns after ~8k (BevAD-M 72.7% SR vs 55.3% on SimLingo data only). DiffusionDrive and Hydra-NeXt (A22) are further precedents. **Why for us:** two-way obstacles need a committed stay-or-pass choice (A41 gates two heads; a
generative head samples one mode instead of averaging), and our data will grow 5-8x (A56, A46) into the range where point estimators saturate.
- **Design:** flow matching (or a 3-mode MDN with winner-takes-all as the cheap version) over the path offsets and speed bins, conditioned on the policy token; 4-8 steps; decode with anchors (DiffusionDrive-style) or the median of 5 samples so that run-to-run noise does not rise; compare at equal data first
  (expect +2-4 SR points, below A52's MDE: judge by overtaking success and the two-ways families), then at 3x data.
- **Readouts:** overtaking success, the three two-ways families, per-route run-to-run SD of DS (must not rise). **Single camera:** yes. **Cost:** head-only; the head's inference grows with the number of steps.

### A56. SimLingo-Data pilot: Dreamer counterfactual labels, bucket index and augmented views on our camera - **Next** (Tier 2; supersedes the A37 step 0 label pass) (S-122)
**Facts** (`scripts/analysis/simlingo_dreamer_probe.py`; `E:\MThesis_EXP\analysis_20261004\simlingo_dreamer_probe_0410.txt`): `RenzKa/simlingo`, 1.17 TB, 3,308,315 frames, 38 scenarios, PDM-Lite; **the same camera as `PDM_Lite_Carla_LB2`** (1024x512, FOV 110, x = -1.5, z = 2.0; confirmed in the repo's
`team_code/config.py`) plus an **augmented-offset RGB per frame** (random shift and yaw: recovery views); measurements / boxes JSON in the PDM-Lite layout we already load; driving data ~580 GB for training (1-scenario routes 274 GB in 12 chunks, 3-scenario routes 202 GB in 16, LB1-split
72 GB, parking-lane Town12 31 GB; the ~580 GB validation split has the same format). **Dreamer labels (~9 GB in total):** per frame, counterfactual alternatives with 10 waypoints, 20 route points and a kinematic roll-out verdict: `faster` (throttle ramp to ~14.8 m/s) crashes into a
dynamic actor in **60.5%** of the 23,091 frames of one archive, `stop` 46.6%, random `target_speed` 43.2%, `faster_factor` 10.7%, `slower` 4.3%, `slower_factor` 0.2%, `crash` (steer at a named vehicle: id, distance, type) 96.2%, `lane_change` left / right with `allowed` (sidewalk, oncoming lane:
41% not allowed). **`buckets_paths.pkl` (620 MB, no import opcodes):** the CarLLaVA / SimLingo sampling buckets, 42 over 2.92M frames (see A38).
- **Use:** (1) A37's labels without our own label pass: a per-frame "speed v crashes" critic from the `target_speed` / `faster` / `stop` entries, AUROC first; (2) A38's hazard sampler = the buckets; (3) more same-camera data for A49 (lead-vehicle families) and a small A46; (4) the augmented views for the recovery question (S-079).
- **Pilot:** one data chunk (~23 GB; stream-extract only rgb + measurements + boxes, skip LiDAR) and its Dreamer chunk; ~100k frames of features on one GPU (~6 min at 300 fps); train the critic head; AUROC by stratum (vehicle / walker / none, distance); then the closed-loop decode of A37.
- **Licence (Wayve, non-commercial):** academic research is allowed; attribution is required in derivative work; models trained on it may be published only on an academic basis and free of charge; **no use in the operation of a vehicle or robot**; revocable; indemnity clause. Simulation research is the intended use (SimLingo's own),
  but check the clauses with the supervisor before the thesis relies on it; the HF relay stays private.
- **Caveats:** a separate collection from `PDM_Lite_Carla_LB2` (random weather; the README says the frames are not from unique routes); chunk files mix scenario types, so a hazard subset still costs whole ~23 GB chunks; Dreamer's crash test is open-loop against recorded actor futures (the A37 assumption).
- **Single camera:** yes. **Cost:** ~25 GB of downloads + ~9 GB labels for the pilot; a box with 150 GB disk and one GPU.

### A57. Newsvendor decode: a low speed quantile that relaxes with standing time - **Next** (eval-only; Tier 1; with the matched-speed control) (S-122)
**Status 2026-10-05 (S-126): code done** (`WOR_NEWSVENDOR="q0=0.3;T=8;qmax=0.7"`: the quantile rises q0 -> qmax over T s of standing, then the plain mean; the literal q -> 1 would read the fastest bin with mass); matched-speed control `WOR_SPEED_SCALE=0.9` (A52). Job: `CKE:<label>:<ckpt>:<routes>:WOR_NEWSVENDOR=q0=0.3;T=8;qmax=0.7`.
For an asymmetric cost (c_o per m/s above the right speed, c_u per m/s below) the optimal point forecast is the quantile q = c_u / (c_o + c_u) of the predictive distribution, not its mean. Our speed posterior (two-hot over 8 bins) is decoded by its mean; a static median deadlocked (A28, -22.9 DS: a
stop / go posterior is bimodal and the median snaps to the stop mode) and so did the hard brake (S-059). A collision multiplies the score by 0.6 while waiting costs little until the 200 s cap, so the right speed is a *low* quantile that **rises as the car has waited**: the cost of waiting accumulates.
- **Design:** v = Q_q(posterior) with q(t) = q_0 + (1 - q_0) min(1, t_standing / T); start conservative (q_0 in {0.2, 0.3, 0.4}), relax to the mean-like decode with T in {4, 8} s; reset when the speed is above 3 m/s. It shares the stall breaker with A50 (the creep clamp is the q_0 -> 0 limit).
- **Control (A52):** the plain head at a global speed scale with the same mean speed. **Success:** fewer collision events per run than the control at the same time-cap share (guard +3 points); abilities Emergency_Brake and Merging.
- **Single camera:** yes. **Cost:** eval-only, ~45 min per config on 12 lanes.

### A58. More routes: Fail2Drive (and Town13 validation routes, Longest6 v2) next to the 220 - **Next** (Tier 0; evaluation infrastructure) (S-122)
**Facts 2026-10-05 (S-126):** Fail2Drive is run with **its own simulator build**: `fail2drive_simulator.tar.gz` on HF `SimonGer/fail2drive` (**12.8 GB**, CARLA 0.9.15 with 30 novel assets: animals, visual noise, adversarial obstacles), the route files are `fail2drive_split/*.xml` (one file per route, e.g. `Generalization_PedestriansOnRoad_1085.xml`) from the GitHub repo, evaluated with `leaderboard/leaderboard/leaderboard_evaluator_local.py` and scored with `tools/f2d_result_parser.py`. So it is a **separate lane type** (their CARLA + their evaluator, our agent through the standard Leaderboard agent interface), not a route file for our Bench2Drive harness: budget a day, one box, ~13 GB download (3 min at 70 MB/s) + the repo; do it after wave 1.
Fail2Drive (`autonomousvision/fail2drive`, MIT; arXiv 2604.08535): 100 *pairs* of routes = 200 routes in Town13; each shifted route (17 unseen long-tail scenario classes: appearance, layout, behaviour and robustness shifts) is paired with an in-distribution route on the same road, spawn points and traffic, so the paired difference
measures the sensitivity to the shift; the repository ships the PDM-Lite expert and TransFuser++ as baseline agents; SOTA models lose 22.8% success on average.
- **Why:** 220 + 200 routes lower the smallest resolvable difference from ~4.9 to ~3.4 DS at one run per arm (null SD 25 per route); the pairing removes route difficulty; and the shifted classes (layout, behaviour) are what our obstacle and lead families probe.
- **Do:** wire the route files into the harness (same XML family), run E, J and WoR once (WoR keeps its 4 cameras), report in-distribution vs shifted per arm and the gap. **Cost:** about one 220-route-equivalent evaluation per arm (~2 h on 12 lanes).

### A59. Pooled-token feature cache for head-only training - **Next** (Tier 0; a day of code; verify in the first 30 minutes of a box) (S-125)
**Status 2026-10-05 (S-126): DONE and verified on real data.** Parity on 128 real frames: fp32-built cache vs fp32 live max |dwaypoint| 0.0082 m (noise floor 0.0063); step time 0.087 s vs 2.08 s (A40 shared with CARLA lanes); build 101-145 frames/s per shared GPU. Builder default fp32 (`--amp` for bf16: 0.125 m max, bf16's own noise); `--frozen_backbone` takes the run's own backbone. Train with `train_wor.py --pooled_cache <prefix> --vision_grid 4` (frozen backbone, no colour aug, qwen head). Next: an end-to-end training smoke on box 4 and the full-data build.
**Why:** a head-only epoch on 427k frames costs 17.6-19.3 min (3090 / A40, 288x768) because every step decodes 256 JPEGs, paints the route overlay on 24 workers and runs the frozen encoder (50.5% of a Qwen step by the cache script's own measurement, probably a larger share at 288x768 because the head does not grow with the image). `build_feature_cache.py` stores the full 1512x9x24 map per frame (653 KB, 279 GB for J's data), which is why no arm used it.
**Exact for the J family:** `QwenWorldOnRailsPolicy` applies `AdaptiveAvgPool2d((4, 4))` to the encoder map first (`vision_pool`), then appends the constant ray-geometry channels and projects (`vision_proj`): nothing learnable sits before the pooling. Arm J's `run_config.json`: `color_aug_prob 0.0`, `freeze_backbone 1`, `use_augmented_camera 1` (stored recovery frames, not a random transform),
`route_overlay 1`, `route_key route_original`, `vision_grid 4`: nothing random upstream of the encoder. As `build_feature_cache.py` argues, this is memoisation (the fp16 storage roundtrip is finer than the bf16 autocast it replaces).
- **Design:** `scripts/training/build_pooled_cache.py` -> one fp16 memmap (N, 1512, 4, 4) = 48 KB per frame, 20 GB for 427k frames (fits the page cache of a 64 GB box) plus the frame index; a dataset mode that returns the pooled tokens and the labels (no JPEG, no overlay, no encoder); one cache per (img_size, overlay, route_key, augmented-views) combination, built on both GPUs of a box
  (~10-17 min, the cost of one epoch). Parity test: head output and loss on 256 frames, live encoder vs cache, within the fp16 tolerance. History tokens (A48) become an index shift.
- **Not covered:** arms that change the pooling grid (A54 space-to-depth, A31, A42 pyramid queries) or the pixels (A17 overlay variants, A25 IDM overlay, A35 self-prompt) need their own cache or the live encoder.
- **Expected gain (a projection, measure first):** epoch 17.6 -> ~2-4 min; a 5-epoch fine-tune ~15 min instead of 1.5 h; 3 trained seeds per finalist ~45 min (A52: 26% of a run's variance is the training seed). **Kill criterion:** < 3x faster. **Single camera:** yes.

### A60. Persistent lane queue for the CARLA evaluation harness - **Next** (Tier 0; 1-1.5 days of code + a 40-route parity check) (S-125)
**Revised 2026-10-05 (S-126): not a persistent evaluator.** The Bench2Drive fork starts its own CARLA per launch and the vanilla evaluator would break comparability with every earlier number. The measured waste is start-up (78 s) + hand-over (43 s) per *job*; jobs of 8-9 routes (`scripts/ops/gen_lanes_a36.py`) cut it to ~13 s per route against ~45 s at 2.7 routes per job. Do: >= 8 routes per (arm, lane) job, trim `sleep 120` / the 60 s guardian poll (not while lanes run: bash re-reads a running script), balance by route wall time, **lanes <= VRAM / 5.5 GB** (S-126: 8 lanes on 24 GB = OOM). The gap between routes is 86 s median (teardown + world load + scenario + agent set-up); its split is still unknown (an OOM-free single lane with timestamps would give it).
**Why (measured, `box_rates.py`):** over 601 lane jobs / 1,606 routes / 186 lane-hours on the six boxes of 10-01..10-04 a lane spends **47% on the route itself, 29% on per-route overhead, 9% on job start-up (median 78 s: CARLA cold start + agent import) and 14% on hand-over (median 43 s: `sleep 120`, guardian polling, port clearing in `queue_A9c.sh`)**; a job carries 2.7 routes
(a chunk per arm per lane, plus retry jobs). A lane spends 417 s per route-run, 197 s of it the route: 12 always-busy lanes give 104 runs/h, which is what the boxes measured (65-128).
- **Design:** one CARLA server + one evaluator per lane for the whole session, pulling (arm, route) jobs from a shared file queue (atomic claim); the arm's checkpoint and decode flags are read per route so there is no restart between arms; results append to the per-lane JSON as today (`--resume` semantics kept); infrastructure failures are re-queued, driving outcomes never (as `run_leaderboard_resilient.sh`).
  Balanced by construction: no tail where eleven lanes wait for one long route.
- **Expected gain:** start-up and hand-over disappear: 417 -> ~320 lane-seconds per run, ~134 runs/h (+30%). The per-route overhead (110 s on Town12/13, 62 s elsewhere) is unknown in composition: timestamp one lane (world load, scenario build, agent set-up, teardown) in the smoke session; if the agent's model load is a large part, load once per process. **Do not** skip the per-route world load without a parity run (traffic and layer state could change the measurement).
- **Parity check:** 40 lead + obstacle routes with E e15 on the old harness vs the queue: DS difference within the run-to-run noise (SD 8 per route-run for E), same status mix. **Single camera:** n/a. **Cost:** saves ~25 box-hours (~$15-25) over the plan and the hand-over failure modes of per-job relaunches.

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
- **RoG-DAgger (arXiv 2608.24525, August 2026, base SimLingo; S-122):** short-horizon kinematic rollouts build expert demonstrations in safety-critical states, takeover is timed near estimated points of no return, and expert decisions are aligned with the student's field of view: Bench2Drive +5.3 DS / +6.2 SR points, Longest6 v2 22 -> 44 DS, Fail2Drive SR 55 -> 66%.

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

### A21. Remove expert-learner asymmetry from the labels - **Next** (promoted 2026-10-04, S-117: LEAD's central ingredient; Tier 2, with A38)
LEAD: PDM-Lite reacts to actors the cameras can't see and to exact velocities of other cars, which gives
non-causal or "successful but dangerous" demonstrations. Restricting the expert to camera-visible actors:
+1.37 DS. Without re-collecting data: down-weight frames where the expert's hazard/brake is caused by an
actor outside our front camera's field of view (from the logged `vehicle_affecting_id` / `speed_reduced_by_obj_*`
fields plus the camera frustum). Feeds A11.
**Design (S-117).** PDM-Lite logs per frame which actor set the expert's speed (`speed_reduced_by_obj_type`, `_id`, `_distance`; `vehicle_affecting_id`, `walker_affecting_id`; checked in
`carla_garage/team_code/autopilot.py`) and the boxes (`id`, `class`, `position`, `extent`, `yaw`, `speed`, `brake`, `distance`, `num_points`; verified on real archives with `scripts/analysis/pdm_lite_range_probe.py`). The logs have no camera visible-pixel count: visibility = a field-of-view
test on the box position (front camera +-55 degrees) plus `num_points` (LiDAR points in the box) as an occlusion proxy. Relabel offline, one change per arm: (a) frames whose reduction is
caused by an actor outside the front camera's field of view or fully occluded -> weight 0, or relabel with the conservative IDM of the visible actors only; (b) traffic-light reasoning outside
the frustum -> weight 0; (c) conservative braking near visible hazards: IDM headway T and gap s0 scaled by 1.5 for visible lead vehicles; (d) enlarged boxes at unprotected turns. The share of
frames each rule touches is the step 0 number (A45 strata). **First real-log numbers (2026-10-04, Town12, 6 routes per archive):** in the lead-vehicle families the actor that sets the expert's speed is almost always
in the front view (96-99%; outside +-55 degrees 0-4%) and well covered by LiDAR (`num_points` < 10 in 0-5% of the causing actors; 26% in DynamicObjectCrossing), so rule (a) (out-of-view actors) touches few frames
there; what the camera cannot do is *measure* the expert's tight following gap (median 4.0 m on HardBreakRoute, 6.0 m on HighwayExit, 6.6 m on MergerIntoSlowTrafficV2: PDM-Lite's s0 = 4 m + T = 0.25 s) and the
lead's closing speed: LEAD's *uncertainty* asymmetry, i.e. rule (c) (a larger headway for visible lead vehicles) and A38, is the lever for these families; rule (a) matters more at junctions and side hazards. LEAD gets +1.37 DS on Bench2Drive and +11 on Longest6 v2 by changing the expert at collection time; here we change the labels of
the logs we have, and A46 is the same idea with LEAD's own data. The arms share the label pass with A38.

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

### B35. Trunk screen at 30k env steps, 8 seeds per trunk - **Done: no trunk difference at 30k** (S-111 plan B, S-112, S-113)
2026-10-02 (S-113): all 16 runs scored at env30000 (upd28004). GTrXL s0-s7 162.8, 30.7, 0.0, 319.9, 85.6, 53.0, 46.0, 103.8;
ResNet s0-s7 43.0, 172.9, 23.6, 74.4, 0.0, 250.8, 45.0, 153.1. Paired GTrXL - ResNet +4.9 [-87.2, +98.5], 4/8. Seed spread
(0-320 at one 10-episode eval) dominates; a trunk claim on Breakout needs more episodes per eval (B2) or more games (B6).
2026-10-01 (S-112, box G): metric = the env30000 checkpoint at upd28004 (what the inline runs scored). Done: ResNet s4 0.0,
s5 250.8; GTrXL s6, s7 reached 30k, checkpoints on HF `b35_20261001/` (score with `eval_ez_checkpoints.py`). Resume from HF
`b35_20261001_resume/` (checkpoint_latest + replay_latest): GTrXL s1, s2, s4, s5 at 20k. Not started: ResNet s6, s7.
Deferred eval verified bit-exact vs inline. One GTrXL per GPU (two per 3090 ran at ~57 upd/min, ~9 h per run).
Replaces more 100k seeds (B4): 3 seeds per trunk at 100k cannot separate the trunks unless the gap is very large, and Breakout
is one game. A 30k run is an exact prefix of the 100k run with the same seed (`--schedule-steps 100000`), so the env30000
evals already on E: count: GTrXL s0 162.8, s3 319.9; ResNet s0 43.0, s1 172.9, s2 23.6, s3 74.4. To run: GTrXL s1, s2, s4-s7 and
ResNet s4-s7 (10 runs). Evals run in their own process (`--eval-mode deferred` + `eval_ez_checkpoints.py --watch`; the
1-2 h inline evals blocked training, S-111): `scripts/atari/run_screen_30k.sh GPU TRUNK:SEED ...`. Metric: env30000 eval
(10 episodes), compared per seed (rank / bootstrap). Box: 4x 3090-class, 2 runs per GPU (GPU-bound when shared).

### B4. Scale the budget: GTrXL + uniform to 100k - **Running** (GTrXL box Z, ResNet box Y, S-093)
Atari-100k is the standard benchmark; EZ-V2's own run here gives 321.2 at 100k (paper 400.1;
curve 9.8 / 37.2 / 92.0 / 278.2 / 363.1 / ... at 10k-50k, S-058). One seed each of GTrXL and ResNet,
uniform replay, checkpoint and eval every 10k so the curve can be set against EZ-V2's. ~14 h per run
(a 10k run takes ~80-90 min on a 4070 Ti S); two runs fit on one 24 GB GPU (box T ran two streams in
17.3 GB). Needs a box without an 8-hour cap. Include `replay_latest.npz` in the backup sync: without
it a lost box restarts a 14 h run from zero (S-064, `tried_and_ruled_out.md`); check its size against
the box's bandwidth price first. Run alongside B3, not after it.

### B3. Diff our prioritized replay against EZ-V2's - **Screen done (S-116): alpha = beta = 1 on the fixed code gives 16.2 / 12.8 vs uniform 15.0 / 15.3 - no gain, no break (2 seeds, 10k)**
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

### B2. An evaluation that can carry the claim - **Done for the deterministic protocol (S-116): GTrXL 105.9 vs ResNet 98.6, +7.2 [-79, +98]; sticky-action column still to do**
Both harnesses repeat episodes: ours (10 episodes, deterministic search, 1-30 no-op starts, e.g.
[57, 57, 57, 33, 57, ...]) and EZ-V2's own (seed 0's 30 episodes gave only 3 distinct scores: 5, 6,
13; S-086). Re-evaluate every final checkpoint of both (our 12 ResNet/GTrXL 10k finals, EZ-V2 seeds
0-4) with 30+ episodes and sticky actions (p = 0.25) in **one** harness, so the comparison with
EZ-V2 is like for like.
**Result 2026-10-03 (S-116, 16 B35 checkpoints, 30 episodes, no sticky, 16 sims):** GTrXL 105.9 / ResNet 98.6, +7.2 [-78.7, +97.7],
permutation p 0.89; tunnel rate 19% vs 15%. **The 30 episodes are ~4 independent games per checkpoint** (4 distinct scores): n = seeds.
Sticky actions (p 0.25) give 21-27 distinct scores and ~1/3 of the deterministic score. To do: the sticky column for all 16
(`eval_ez_checkpoints.py --sticky 0.25 --graph`, 5 of 16 done), now cheap with B23's graphed search.
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

### B36. Make the GTrXL mixer actually train (weight-decay exemption) - **Next** (S-116; the highest-value Atari item)
**Finding (2026-10-03):** all mixer weights shrink by x0.13 per 10k updates under SGD lr 0.2, wd 1e-4, momentum 0.9 (|Wq| 6.5 -> 0.026
at 30k, 0 at 40k, identical in all 8 seeds; attention uniform, gates at 0.5, only `out.bias` alive). B1/B35/B4/B2 therefore compared
ResNet with ResNet. **Test:** `train_ez_offpolicy.py --mixer-weight-decay 0` (flag exists), 30k screen, 8 seeds, vs the 8 ResNet seeds of B35
(and vs the dead-mixer GTrXL, already on E:); check |Wq| grows and attention entropy < 1 before looking at scores. If the mixer still
does not train with wd 0 (zero-init `out` starves it of gradient), next: non-zero `out` init or AdamW for the mixer (lr ~1e-3), QK-norm
(B18). Everything transformer-specific (B5, B18-B20, B26, B27, B29) needs this first. A 5000-step control on box Q (S-116) shows the
mixer stops decaying with wd 0 (73.34 -> 73.40) and the output projection grows (0 -> 1.35 / 2.20), but q/k/v/FFN are still at their
init values at 5k (closed gate, zero-init output): the 30k screen has to show whether they train.

### B37. Make the transformer mixer train: AdamW parameter group, open gates, and a module-update audit - **Next** (the B36 follow-up; highest-value Atari item) (S-118)
**Result 2026-10-05 (S-127): the AdamW 3e-4 mixer recipe FAILED on all four seeds (30k screens).** Every eval at env 5k / 10k / 15k / 20k / 25k / 30k is 0.00 (10 episodes), except seed 2 at env 5k (1.40). The audit shows the mixer weights growing without bound (|dW|/|W0| of the `io` and `norm` tensors ~1e12-1e13, |mixer W| 318-1234 by env 13k) while the attention/FFN blocks move 4-22x; seed 2 diverged (gradient norm ~3e9, value loss 10-18, reward loss 15-27 from env 21k), seed 0's policy loss fell to 0.001 at env 13k (collapse), seed 3 died at env ~22k with a CUDA device-side assertion after a resume. Cause not isolated; AdamW lr 3e-4 on this mixer with the gate init is unstable. Next try: mixer lr 3e-5 with gradient clipping (`--mixer-grad-clip`) and the audit on, 10k-step curves first. Note the first launch also ran SGD for 20 minutes because `run_b36.sh` was patched after shipping (see `verify-launch-flags-and-ship-all-patched-files`).
**Step 1 done 2026-10-05 (S-126):** probe on box 2, 2,000 updates per setting: AdamW on the mixer moves q/k/v by 7% / 21% / 65% of their init norm at 250 updates (lr 1e-4 / 3e-4 / 1e-3; pass > 5%) where SGD stayed at init; losses in the SGD baseline's range. **Step 2 running** (AdamW 3e-4, wd 0.05, 500 warm-up, mixer clip 1.0, 4 seeds x 30k steps, box 2, done ~13:05 Athens; `--audit-every 1000` prints `[B37 audit]` lines: relative weight change and gradient norm per category). A first launch ran plain SGD by mistake (20 min lost, S-126).
**Evidence (2026-10-04, B36 on box 3):** with `--mixer-weight-decay 0` the mixer stops decaying (|W| 73.34 -> 73.40), but the blocks do not learn: seed 2 at 15k has |Wq| 6.510 (init
6.5), FFN gate 13.069, GRU gate bias 2.000, attention entropy 0.99 of the maximum, while the output projection grows (|out| 0.96 -> 2.49). A non-zero output init (`--mixer-out-init-std
0.02`) does not change it: |Wq| 6.507 at 10k, |out| 2.63. The gate starts closed (z ~0.13) and the block output is multiplied by the small output projection, so q/k/v/FFN receive a
gradient orders of magnitude below the rest of the network, and SGD moves a weight by lr x gradient. **Scale-invariant optimisers (Adam) are the standard fix for exactly this mismatch,
and every transformer is trained with one.** Everything transformer-specific (B5, B18-B20, B26, B27, B29) is blocked on this.
- **Step 0 (done offline, S-117):** `scripts/analysis/b37_weight_trajectory.py` on the 100k runs of seed 0 (10 checkpoints each): every conv/head module of both trunks changes by
  ~0.1-1.2 of its norm per 10k updates (the SGD recipe keeps them moving); the two mixers' change equals the weight-decay rate (0.86 per 10k, predicted 1 - 0.13 = 0.87) and their norm
  falls 15.4 -> 0.7: **no other module of the port is dead**. Output on E: `analysis_20261004/b37_traj_*.json`.
- **Step 1, a 10-minute probe (no scores, graph search):** 2,000-update runs logging, per module, the gradient norm and the relative update |dW|/|W| every 250 updates, with
  (a) `--mixer-optimizer adamw --mixer-lr {1e-4, 3e-4, 1e-3} --mixer-wd 0.05`, betas (0.9, 0.95), 500-update warm-up, mixer-only grad clip 1.0 (conv trunk and heads stay SGD, EZ-V2's recipe);
  (b) open gates: GRU gate bias init 0 instead of 2 (zero-init output keeps the identity start); (c) a + b; (d) a ReZero/LayerScale scalar on each residual (init 0, lr x100).
  **Pass:** |dWq|/|Wq| > 5% and attention entropy < 0.97 by 2k updates, with the first 2k updates' losses within noise of the SGD run.
- **Step 2:** the winning setting on the B36 30k screen, 4 seeds, against the ResNet and the wd-0 GTrXL runs already on E:; read |Wq|, entropy and gate opening at 5k / 10k / 20k before the
  scores (`b36_diag_watch.sh` does this).
- **Kill:** no setting moves Wq in the probe -> the mixer design (not the optimiser) is the problem: try the plain pre-LN residual block without the GRU gate (B5's "GTrXL without gating").
- **Consistency and prior art (S-118):** the CARLA head (AdamW, lr 3e-4) needed exactly this class of fix, a weight-decay exemption for its gates and norms (ch. 13.6c); EZ-V2 itself trains
  Atari with SGD but proprioceptive control with Adam; training transformers with Adam beats SGD in general (heavy-tailed gradient noise). The PPO-era "gates frozen at init" note (S-035-S-037)
  had another root cause (the Kaiming init of the Impala encoder, S-043) and is not evidence about the optimiser.
- **Cost:** probe ~10 min per setting on one 3090 with the graphed search; step 2 ~4.5 h per 30k run with one run per GPU.

### B38. Pessimistic (and optimistic) search from two-network disagreement - **Next** (eval-only step 0; novel for an EZ-V2 search) (S-118)
**Evidence:** B24: 64 simulations instead of 16 lowered the score on 4 of the 6 checkpoints finished (307.5 -> 158.2, 157.3 -> 74.9, 159.7 -> 107.7; one rose 47.6 -> 94.4). The model's
imagined values at depth are over-optimistic: the search picks the actions whose errors are most favourable (model exploitation / winner's curse). Model-based RL counters it with
pessimism toward uncertain model predictions (MOPO, MOReL, ensembles in PETS/MBPO) and value-based RL with the min of two critics (clipped double-Q). We have a second network at no
training cost: every checkpoint stores `target` next to `model`.
- **Step 0 (eval-only, graph search; two model calls per simulation):** keep two latent states per tree node (online and target network, each rolled through its own dynamics); the leaf
  value and reward are v = mean(v_on, v_tg) - lambda * |v_on - v_tg| / 2, lambda in {0, 0.5, 1}. Evaluate B24's 16 checkpoints at 16 / 32 / 64 sims, 30 episodes. **Success:** the score
  is non-decreasing in sims on >= 12 of 16 checkpoints, and the 64-sim mean >= the 16-sim mean (today 64 sims is lower). One box-hour on 4 GPUs.
- **Step 1 (training):** the same rule in the reanalyze targets, with an EMA target (`--target-ema 0.99`) as the second member, since a 200-update hard copy is almost the online network
  right after a copy.
- **Step 2 (exploration, Breakout's tunnel regime is bimodal, B2):** optimism at acting time only: Q_root + beta * disagreement, so the agent searches where the models disagree while the
  training targets stay pessimistic.
- **Kill:** no change in the slope of score vs sims, or the 16-sim score drops by more than 10%.
- **Differs from** B24 (measures the slope), B30/V-MCTS (when to search), B34 (target estimators): none uses model disagreement inside the search.
- **Prior art (S-118):** Epistemic MCTS (ICLR 2025) propagates epistemic uncertainty through an AlphaZero/MuZero tree for deep exploration in sparse-reward tasks (no Atari); Model-Value
  Inconsistency (Filos et al., ICML 2022) builds an "implicit value ensemble" from **one** model and one value function rolled to different depths and uses the disagreement as an uncertainty
  signal for pessimism; a TU Delft thesis uses bootstrapped ensembles for MuZero exploration. We found no use of either signal inside a Gumbel / EZ-V2 search.
- **Step 0b (no second network):** inside the existing tree, a node's inconsistency = |V_head(s) - (r + gamma * backed-up child value)|; penalise the backup by lambda * inconsistency; same
  B24 protocol. Needs no extra model calls, so it can also be switched on in training.

### B39. Decouple the search depth of reanalyze from acting, and use a fresh network for targets - **Next** (one flag; the graphed search makes it affordable) (S-118)
Reanalyze is ~62-71% of training time at 16 simulations (time split c/t/u ~6/71/23 eager, 2/62/36 with the graphed search on a 3090 at batch 256) and it generates every policy and value target, while acting needs only
a good move. Today both use 16 simulations and the reanalyze network is the target copy up to 200 updates old.
- **Flags:** `--reanalyze-sims N` (acting stays at `--num-simulations`) and `--reanalyze-net online|target`.
- **Arms (ResNet, 10k, 4 seeds each, S058f recipe):** 16/16 control, 32/16, 64/16, and 16/16 with the online network for targets; per-update cost of reanalyze 2x / 4x, partly paid back by
  the graph (5.4-5.5x at 32/64 sims).
- **Readouts:** 10k score; value-target error against Monte-Carlo returns on a fixed held-out state set (the reanalyzed value should be closer); policy-target entropy.
- **Caveat:** B24's deeper test-time search hurt, so deeper targets may too; run it next to B38's pessimism, which is the companion.

### B40. Train under sticky actions and report both protocols - **Next** (robustness, and evaluations that mean something) (S-118)
**Evidence:** B2: with deterministic evals a checkpoint yields ~4 distinct games however many episodes are run, so n = seeds; with sticky actions (p = 0.25) 21-27 of 30 games are
distinct, but the scores fall to ~1/3 (GTrXL s3 307 -> 103, ResNet s5 254 -> 88): the agents rely on deterministic dynamics. Train with `repeat_action_probability = 0.25` (the wrapper
exists, `make_atari_env`; add `--sticky-train`) and evaluate under both protocols.
- **Arms:** ResNet and GTrXL (after B37) at 30k, 4 seeds each, plus a deterministic-trained control already on E:.
- **Questions:** does the learned model and search degrade gracefully; does the transformer trunk help more when the dynamics are stochastic (B5's attribution); how many distinct games
  per checkpoint does a sticky eval give (tunnel rate and median become usable per seed).
- **Caveat:** not comparable with EZ-V2's published numbers (no sticky actions in the Atari-100k protocol); report as a robustness row next to the standard one.

### B41. Privileged-state auxiliary (ALE RAM) as an upper-bound probe of the representation - **Later** (diagnostic; not for the headline comparison) (S-118)
Aux heads on the latent regress the ball position and velocity, the paddle position and the brick count, read from the emulator RAM (the AtariARI annotation table for Breakout, Anand
et al. 2019); the labels exist only in training. Same recipe as the CARLA rule (privileged information as targets, never as inputs), but here it breaks comparability with EZ-V2.
- **Question:** is representation learning the bottleneck at 10k and 30k? If the score jumps (10k: 15 -> 25+), object-centric inputs (B14) are the right investment; if not, search,
  dynamics or optimisation (B37, B38) are.
- **Also:** probing accuracy of the *unaided* latent is B28's first measurement (does it already encode the ball?).
- **Cost:** one flag, a RAM read in the env wrapper, 2 seeds x 2 budgets.

### B42. Multi-horizon value heads as an auxiliary loss - **Later** (cheap, one flag) (S-118)
Predict the return-to-go at several discounts (gamma in {0.9, 0.97, 0.997}) from the same latent, as auxiliary targets next to the main value head (Fedus et al. 2019,
"Hyperbolic discounting and learning over multiple horizons", reports gains on Atari from the auxiliary horizons *(unverified: abstract only)*). BBF anneals gamma 0.97 -> 0.997 (B8) but nobody here tried the auxiliary
form, which shapes the representation without changing the control problem. Targets from the same reanalyzed values (no extra search). Arms: 10k, ResNet, 4 seeds; kill if
the main-head value error or the score does not move.

### B43. Depth-capped search and the model's error against imagined depth - **Next** (eval-only; explains B24; Tier 1) (S-118)
**Step 1 code done 2026-10-05 (S-126):** `GumbelMCTS(max_depth=D)` (a node at the cap is not expanded, it is backed up again with its own value-head output; exact when the cap is loose, identical in the sync-free graph mode, tests), `eval_ez_checkpoints.py --max-depth`, `scripts/ops/run_b43_depth.sh "3 5 8 0" NPAR` for the 16 restored B35 checkpoints. Not run yet (no free GPU today).
**Evidence:** training unrolls the dynamics for K = 5 steps, but at 64 simulations the tree reaches depth 17 (16 sims: 6, 32 sims: 12; S-116 tree-depth measurement on real states), so the search uses
the model well outside the horizon it was trained on. B24 found deeper search lowering the score on 4 of 6 checkpoints, and S-058 found the value-prefix head miscalibrated at imagined depth 1.
- **Step 0, error curve (offline; `scripts/analysis/b43_model_drift.py`; nine checkpoints done 2026-10-04, S-121):** the script rolls the model forward with the real actions on real replay trajectories and compares each
  imagined step k with what the same model outputs on the real observation at t + k (latent cosine in the SimSiam space, value, policy KL, value prefix); 384 start states, depth 12, laptop CPU. **Result** (5 GTrXL: B35 s1, s2, s4, s5,
  B36 s2; 4 ResNet: B35 s4-s7, states from the GTrXL s1 buffer because the ResNet buffers were not saved): GTrXL latent error 1-cos 0.010-0.019 (depth 1) -> 0.027-0.039 (depth 5) -> 0.10-0.21 (depth 12), **the per-step increase grows
  1.7-3.3x after depth 5 (median 1.9)**; **ResNet's per-step error is ~3x higher from the start and then constant (0.9-1.3x): linear, no acceleration**. **In all nine the policy KL rises 2.0-4.0x (mean over depths 6-12 vs 1-5) and the
  argmax agreement with the real-state policy falls to 0.62-0.84 at depth 12**; imagined values drift low (ResNet s5 -0.10 at depth 12), so the damage is variance and the winner's curse, not mean optimism (B38's target). The earlier
  two-checkpoint reading ("doubles in both trunks") is corrected: the latent acceleration is GTrXL-only (and these GTrXL have dead mixers, S-116); the degradation of the policy and value that guide the tree is general. One replay per
  trunk, no link to B24's scores yet. Next: correlate the growth after depth 5 with each checkpoint's B24 slope; EZ-V2's own checkpoints are the reference (they predict the prefix accurately, S-058). Outputs:
  `E:\MThesis_EXP\analysis_20261004\b43_drift_*.json`, `b43_summary_0410.txt`.
- **Step 1, cap (eval-only):** `GumbelMCTS(max_depth=D)`: a node at depth D is not expanded further and is scored by its value head; D in {3, 5, 8, unlimited}; B24's 16 checkpoints at 64 sims, 30
  episodes, graph search (a `max_depth` knob next to `depth_bound`). **Success:** the score stops falling with sims at D = 5.
- **Step 2 (training, if step 0 shows the error grows after K):** K = 8 with a lower weight on steps > 5, so deeper search has something to stand on.
- **Forecasting lens (S-123):** direct multi-step heads instead of iterating the dynamics beyond the trained depth: B45.
- **Companion:** B38 (pessimism), which attacks the same compounding error from the value side. **Cost:** step 0 and 1 are eval-only, ~1 box-hour on 4 GPUs.

### B45. Root-anchored direct multi-step heads as the deep-leaf evaluator - **Next** (after B43 step 1; Tier 2; one flag + 4 runs) (S-123)
**Evidence:** the DLinear study (Zeng et al., arXiv 2205.13504) found that the long-horizon accuracy of Transformer forecasters comes from *direct* multi-step output, not from attention; iterated forecasters accumulate error with the horizon. Our dynamics iterate (unroll K = 5 in
training) and B43 / S-121 shows the policy KL and the value error x2-4 at depths 6-12 in nine of nine checkpoints and a supra-linear latent drift for GTrXL.
- **Design:** for a tree node at depth d (action prefix a_0 .. a_{d-1}) evaluate with a *direct* head V_d(s_0, a_{0:d-1}) (the root latent and an embedding of the action sequence -> value / value prefix at that depth), trained on the same replay segments (the real return under the real actions plus the
  bootstrapped value, like multi-step value targets), instead of the recursively imagined latent beyond depth 5. Step 1: auxiliary loss only (B42's multi-horizon heads are the same family). Step 2: in search at depth > 5 combine with the recursive value (the minimum, or a learned gate).
- **Readouts:** B43's curves (value error and policy agreement vs depth) and B24's score vs simulations (should stop falling). **Cost:** B43 step 1 first (a depth cap says whether depth is the problem); then a flag and four runs.

### B46. Event and ball-forecast auxiliary (Event-Aware World Model style) - **Later** (Tier 3; one flag) (S-123)
**Evidence:** the Event-Aware World Model (arXiv 2601.19336) predicts automatically extracted events (segment boundaries of the observation stream) as an auxiliary task: 10-45% over existing MBRL baselines on Atari 100K, Craftax and DMC. Forecasting practice: change-point and time-to-event targets.
Breakout events are few and exact (wall bounce, paddle hit, brick hit, ball lost) and the ball path is piecewise linear with reflections.
- **Design:** auxiliary heads on the latent state: (a) the type of the next event and the number of steps to it (classification + regression), (b) the landing column of the ball at the paddle row; labels from the replay (reward, lives, ball motion; the RAM probe of B41 for the landing column), no extra interaction.
- **Readouts:** 10k / 30k scores against the same seeds without it, probe accuracy, B43's drift curves (do the heads reduce the latent error at depth 6+?). **Cost:** one flag, 2-4 runs.

### B44. How many zero scores are stuck loops? Stagnation-aware evaluation and acting - **Later** (diagnostic first) (S-118)
B2/B24 note that games where the agent never launches the ball (no FIRE after a life loss) run to the 27k-step cap and hold up every evaluation; at 64 sims one GTrXL checkpoint finished in 52 min
against 3 h at 16 sims, so search breaks some of the loops. The eval JSON stores `eval_seconds` but no episode lengths, so the share of zero scores that are loops is unknown (the zero-scoring ResNet s4 took 7,032 s against
265-785 s for ResNet s5-s7; GTrXL evaluations are slower in general, 4,225-7,297 s).
- **Step 0:** log per-episode length, reward-free streak and action histogram in `eval_ez_checkpoints.py`; re-score the B35 / B2 checkpoints and report the share of zero scores that are loops.
- **Step 1, if loops are common:** a generic stagnation rule (no reward, no life change and no screen change for N steps -> sample a random action) in evaluation, reported as a second protocol row
  (it departs from EZ-V2's protocol), and the same rule in the acting search so the loops do not fill the replay with dead data.
- **Why it matters:** the seed SD of 100 that blocks every trunk claim comes largely from 0.0 vs 300 outcomes; removing the loops would shrink it without changing the policy.

### B5. Why does GTrXL beat ResNet? - **Later** (after B4; after B36: the current GTrXL has a dead mixer)
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

### B15. Plasticity diagnostics on the saved checkpoints - **Done (S-116): no rank collapse, dormancy plateaus by 50k - B8/B13's premise not supported at 100k**
Before paying for resets (B8) or normalisation fixes (B13), measure whether our runs lose plasticity at all. For
each saved checkpoint of GTrXL s0 and ResNet s0 (best, 40k, 50k/60k on E:), run a few replay batches through
the model and record:
- the dormant-neuron ratio (tau = 0.025, Sokar et al. 2023) per layer;
- the feature effective rank (srank) of the representation;
- the weight norm, and the gradient norm per layer on a fixed batch.

Rising dormancy or falling rank from 20k to 60k, alongside a flat score, would justify B8/B13. Flat metrics
point elsewhere (search, targets). Log the same metrics in `train_ez_offpolicy.py` at every eval from now on
(cheap: one batch).

### B18. GTrXL seed variance: attention entropy, then QK-norm - **Step 1 done (S-116): entropy is maximal (uniform) because the mixer weights decayed to 0 - see B36; QK-norm is moot until the mixer trains**
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

### B24. Test-time search scaling on saved checkpoints - **Running** (64 sims, box P, 6 of 16 done; restart the rest with `--graph`) (S-115/S-116)
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

### B23. Vectorise: a sync-free, graph-captured search, then several seeds per process - **Step 1 done for evaluation and training (S-116: bit-identical, 5.4-8.8x at 16-64 sims; `--graph-search` in the trainer, 1,600-update run identical to eager); batch-256 box check and step 2 next**
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

### B26. A world model that learns how symmetric each game is (soft, detected symmetry, used in search) - **Next** (novel; step 0 eval-only, code ready: `eval_ez_checkpoints.py --flip-avg`, S-115)
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
**Step 0 (no training, run first, with B24):** step 4 at test time only on the saved s0 checkpoints: flip-averaging on vs off. **Done on 4 B35 checkpoints (S-116, `--flip-avg`): -22, -73, +73, -25 points (mean -12): no consistent gain from mirror-averaging the root.**
Gains on Breakout and losses on an asymmetric game would show the symmetry is exploitable and must be detected, not assumed.
**Games:** a clear mirror (Breakout), vertical + UP/DOWN (Pong), partial/maze (MsPacman or Boxing), a directional negative
control (e.g. RoadRunner). The detector, not these guesses, decides which is which. 30k screen, 3 seeds, then 100k.
**Risk:** gates and detector may not settle within 100k steps; the closed-gate init and step 0 bound it.

### B27. Put the "XL" back, inside the imagination: attention over the imagined unroll - **Later** (novel; medium odds)
"XL" = Transformer-XL (Dai et al. 2019): segment memory over past steps + relative positions. GTrXL (Parisotto et al. 2020)
keeps both and adds pre-LN and GRU gating. **Our trunk has only the gating** (`TokenMixer`: 36 spatial tokens of one state,
learned absolute positions, no memory). B26's offset bias restores the relative-position half. For memory: EZ's
value-prefix head is an LSTM over the *imagined* unroll (`value_prefix`, `ez_model.py`). Replace it with causal gated
attention over the imagined trajectory, so the model's own rollout is the memory. Real-time latent history is UniZero's
(B12), so this variant is the novel one. The value-prefix head is where the port broke before (S-058, S-067): screen for
reward calibration (the S-058 probe), not only score.

### B28. What does the EZ-V2 world model learn, and what does attention add? (mechanistic study) - **Next** (analysis; high odds)
"What model does MuZero learn?" (arXiv 2306.00840) did this for MuZero, not for EZ-V2 or attention trunks. On the saved s0
checkpoints (10k ... 80k): linear probes of the latent for ball/paddle position and velocity; k-step imagined vs real
latent error; GTrXL attention maps against object positions; when the "tunnel" behaviour appears (B2's bimodality).
Turns B5's "why does GTrXL help" into something visible; no training, CPU/GPU-light.
**Uncertainty-aware search (considered, not added as novel):** Epistemic MCTS (ICLR 2025) already propagates epistemic
uncertainty in MuZero-style search. Worth a try only as an application to our phantom-reward failures (S-058).

### B29. Action-routed experts in the dynamics network - **Later** (novel twist on MoE; medium odds; zero extra FLOPs)
Prior work: SoftMoE in deep RL (Obando-Ceron et al., ICML 2024, arXiv 2402.08609) - but "Don't flatten, tokenize!" shows
the gain is tokenisation (one scaled expert matches four), which GTrXL already has; MoE world models route by **task**
(Mixture-of-World Models, arXiv 2602.01270; ScaleZero), multi-task only. **Idea:** one small expert FFN per discrete action
in the dynamics transformer block, hard-routed by the chosen action (no router, no balancing loss, one expert runs per
step = today's FLOPs; group experts for 18-action games). Fixes EZ-V2's scalar action plane `a/|A|`, which orders
discrete actions (B19). With B26: tie the LEFT expert to the mirrored RIGHT expert behind a soft untie gate.
Screen: 30k, 3 seeds, Breakout + one 18-action game; also check imagined-reward calibration (S-058 probe).

### B30. Amortised reanalyze: a learned gate on which states need real search - **Later** (efficiency; medium-high odds for speed)
Reanalyze is ~71% of wall-clock and seeds are the bottleneck. A small head predicts the 16-simulation search's improved
policy/value from the network's one-step outputs; run the real search only where the prediction is uncertain or
disagrees (e.g. 30-50% of reanalyzed states). A learned component that *removes* compute. Prior: Thinker (arXiv 2307.14993)
learns the improvement operator (costly); ReZero and V-MCTS cut search cost by rules. The learned gate is to be verified
as open (add to the deep-research prompt). Check the score curve against full reanalyze; report wall-clock per 10k.
**Efficiency rule for B29/B30 and later architecture ideas:** no extra environment steps, <= ~10-20% more wall-clock
(ideally less), no extra seeds needed. Generic MoE and meta-learned search/optimisers (MCTSnets, arXiv 1802.04697; learned
optimisers) fail it at single-task 100k.

### B31-B34. Survivors of the deep-research report (reviewed in `docs/design/deep_research_review_2026-09-29.md`)
- **B31. 2D-axial RoPE in the GTrXL mixer** (cheap arm, with B26): translation-relative attention, ~20 lines, replaces the
  absolute `self.pos`. Uses signed offsets, so it cannot carry B26's mirror symmetry; run it as the "translation only"
  arm next to B26's symmetric offset bias. Prior: RoPE-ViT (ECCV 2024, arXiv 2403.13298).
- **B32. Attention-head disagreement loss** (with B18): penalise cosine similarity between heads' attention maps
  (Li et al. 2018, arXiv 1810.10183), lambda <= 0.05. Target: GTrXL's seed SD. Protocol: (1) intrinsic check - head
  cosine drops within 10k steps (measure our baseline in B18 step 1); (2) Brown-Forsythe on per-game standardised scores,
  4 games x 6 seeds per condition. 6 vs 6 seeds on one game cannot separate SD 11.3 from 6.0 (F = 3.55, p ~ 0.10).
- **B33. Latent transposition table in Gumbel search** (offline probe first): hash expanded latents (cosine LSH on the
  **flattened** latent, not a pooled one, + cosine >= 0.98 verification) and count hits on saved checkpoints' searches.
  Our search: m = 4 root actions, halving 4 -> 2 -> 1, subtrees ~2-4 deep. **Count depth-1 action aliasing first**
  (distinct actions, same next latent, e.g. FIRE ~ NOOP mid-rally): if common, merging aliased root actions frees Gumbel's
  4 slots and is simpler than a transposition table. Build the table only if deeper hits are >= 5%.
- **B34. Retrace(lambda) value targets under the h-transform in reanalyze** (Later): correct targets in raw scale with
  truncated ratios pi_MCTS/mu (store mu), then map back through h. Weigh against EfficientZero's adaptive-horizon
  correction and EZ-V2's search-based value estimation before claiming novelty.
- B29 update: condition the dynamics on the action with **adaLN-Zero** (per-action scale/shift, identity init; DiT found it
  better than cross-attention), not cross-attention.

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
