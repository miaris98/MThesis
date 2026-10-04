# What has worked so far, and what the research says (2026-10-04, S-117 / S-118 / S-120)

Why: before building new algorithms, list what actually moved the score in this project, how strong each piece of evidence is, and
what the published work says about the same problems. Sources are the project's own logs (`challenges/`, `E:\MThesis_EXP`) and a
web check on 2026-10-04 (links at the end). Numbers are DS (Bench2Drive driving score) unless stated; CIs are 95% paired bootstraps over routes.

**Grades.** A = confirmed (CI excludes 0, or >= 3 seeds agree); B = likely (one seed / CI touches 0); C = not established (inside noise);
D = negative (no effect, or it hurt).

## 1. CARLA (one front camera, frozen CARLA-pretrained TF++ backbone, offline imitation of PDM-Lite)

| # | Intervention | Effect | Grade | Source |
|---|---|---|---|---|
| 1 | **Route conditioning** (ego-frame route points + painted overlay) | lateral error was 0.24 m in every configuration = exactly the predict-zero baseline (the data had 100% LANEFOLLOW commands); with the route the policy steers | A (blocker) | ch. 9.4-9.7 |
| 2 | **Transformer head vs WoR's conv head**, same data and backbone (arm H) | H 50.4 vs E 65.4: -15.0 [-25.4, -6.0]; older harness: transformer > conv in 3 of 3 seeds | A | S-090, ch. 13.33 |
| 3 | **E vs the original WoR on the 220 routes** | official 63.4 vs 49.4; paired +13.7 [+8.6, +18.8]; unseen towns +15.9, **train towns +1.3 (tie)**, non-obstacle +20.0 | A | S-108, S-117 |
| 4 | **Obstacle data + `route_original`** (arm J, E e15 fine-tuned 5 epochs) | obstacle routes +7.2 [+0.7, +13.6] vs E (4 seeds); overtaking success 5.7% -> 16-18% (WoR 11.7%) | A | S-109, S-113-S-115 |
| 5 | ... **but J lost the reactive skills**: lead-vehicle routes (merges, cut-ins, hard brake, highway exit; 57 routes) | **-15.2 [-21.5, -9.0] vs E**; overall J - E = -2.6 [-5.9, +0.7]; official 58.8-61.9 (E 63.4) | A | S-117 |
| 6 | Recovery camera (arm A vs B) | A - B = +3.1 [-1.8, +9.5] after repeats; earlier +7 was one coin-flip route (3457) | C | S-062, S-066, S-079 |
| 7 | Colour augmentation (arm D) | +4.2 [-5.7, +12.8] | C | S-079 |
| 8 | 288x768 instead of 192x512 (arm E vs A) | E e15 65.4 vs A 60.6 (E e20 60.6 = A); no paired CI excludes 0; E became the base because it scored best, not because it was shown better | C | S-075, S-079, S-089 |
| 9 | SWA over e11-e20; grad clip 15 | +1.8 [-7.2, +10.5]; +0.3 [-12.7, +12.9] | C | S-078, S-079 |
| 10 | Geometry loss (heading + curvature) | heading error -92%, closed-loop gain 0 at 3 seeds (sign flipped twice) | C/D | ch. 13.29, 13.33 |
| 11 | Head size 10M/30M/100M, `vision_grid` 8, loss re-weighting, training past e15 (6-town arms) | no effect (e20 < e15 for every 6-town arm) | D | ch. 13.24-13.25, S-062 |
| 12 | 8-town data with obstacle scenarios but the *shifted* route as input | still drives into the obstacle | D | S-063 |
| 13 | Hard brake on uncertainty (P(stop) > 0.9); median speed decode | 38.4 vs 65.6 (3 deadlocked routes); -22.9 [-33.5, -12.6] (blocked 15 vs 1) | D | S-059, S-112 |
| 14 | Swerve-frame oversampling (arm K) | no gain over J | C | S-112 |

**Seeds and noise.** Our arms' run-to-run SD per route is 10-20 DS against 8.6 for WoR; the noise is a collision lottery
(each vehicle collision multiplies the route score by 0.6), so repeats and pooled seeds matter more than extra routes (S-100, S-101).

## 2. What the ledger says

1. **The confirmed wins changed what the model is told** (route input, obstacle data with the unshifted route) or *which head reads the
   features*. Nothing that only re-weighted or re-sized the existing head has shown a closed-loop effect (rows 9-11).
2. **Vision is lightly used.** A no-vision MLP on route + speed + command reaches held-out loss 0.7558 against 0.5793 for the transformer
   with vision (lateral error 0.0713 m -> 0.0503 m; ch. 13.30). Evals that accidentally ran on a *random* backbone still scored 50-60 DS (S-072):
   the painted route overlay and the speed token carry most of the score. A0 (blank image) is still open. This is the most important
   unexplained fact: hazards are rare in the data, so a head can minimise its loss without learning to see them.
3. **The objective is mis-budgeted.** 73.5% of the waypoint loss is the longitudinal coordinate, which is ~ speed x dt and is a function of a
   scalar the head already receives; the PID steers from the lateral coordinate and takes speed from the target-speed head, whose loss
   weight is 0.2 and was never swept (ch. 13.26; no sweep found in the logs).
4. **Collisions dominate.** For J, 20.1 of 36.9 lost points per run are vehicle collisions and 47% of runs have one (WoR 22%, but WoR loses
   24.1 points to not finishing). The loss concentrates in lead-vehicle families (HardBreakRoute, HighwayExit, MergerIntoSlowTraffic: 87-98%
   collisions) and pedestrian families (DynamicObjectCrossing 95%).
5. **J is not an overall improvement over E.** J fixed overtaking (+7.2 on obstacle routes) and lost 15 points on lead-vehicle routes; E
   remains the best single model on the 220 routes (63.4 vs 58.8-61.9 official). Both beat WoR (J +10.4 [+5.7, +15.1], E +13.7).
6. **Where we beat WoR.** Entirely on routes that are neither lead-vehicle nor obstacle scenarios (+22.8 / +23.8; junctions, signals, turns,
   pedestrians). On lead-vehicle and obstacle routes our arms are tied with WoR. State this in the write-up.
7. **Position in the field.** TF++ (camera + LiDAR) 80.3 on our 19 routes against ~65 for ours; published 84.2. SimLingo (one front camera,
   VLM) 85.1, LEAD/TFv6 95.0 (camera-only 360 degrees 91.6). Safe2Drive (2026) shows even those collapse on safety scenarios (LEAD 94.7 -> 40.0,
   SimLingo 85.1 -> 41.0), driven by late or absent braking for pedestrians and work zones: braking is the field's weak spot, not only ours.
8. **LEAD's "learner-expert asymmetry" is the right lens** for the next items: *visibility* (the expert reacts to actors the camera cannot
   see), *uncertainty* (the expert brakes with exact velocities; the student must hedge), *intent* (the student does not know the plan).
   Our new items map onto them: visibility A21/A38/A42/A45/A46; uncertainty A37/A39/A43; intent A15/A41.

## 3. Atari (EZ-V2 port, Breakout)

| # | Intervention | Effect | Grade | Source |
|---|---|---|---|---|
| 1 | Copy EZ-V2's recipe (Gumbel search, value prefix, SimSiam, reanalyze, schedule-steps prefix, per-step BN refresh) | port reaches EZ-V2's level: GTrXL 371.6 @70k, ResNet 377.8 @80k (EZ-V2 321.2 @100k, paper 400.1); one seed | B | S-058, S-110 |
| 2 | **Uniform replay** instead of our prioritised replay | 15.0 vs 0.0 at 10k (EZ-V2 itself 9.8); reward head had been miscalibrated (phantom +0.25 at imagined depth 1) | A | S-064, S-067 |
| 3 | Prioritised replay re-implemented to match EZ-V2 | 16.2 / 12.8 vs uniform 15.0 / 15.3: no gain | C | B3 |
| 4 | GTrXL vs ResNet trunk | 10k: 20.9 vs 14.9 (5 seeds, +6.0 [-3.5, +16.4]); 30k +4.9 [-87, +99]; B2 105.9 vs 98.6 | C | S-079, B35, B2 |
| 5 | **The GTrXL mixer was dead**: weights decayed x0.13 per 10k updates under SGD wd 1e-4; with wd 0 the blocks still sit at their init (|Wq| 6.510 at 15k) | rows 4 and the B1/B35/B2/B4 comparisons compared ResNet with ResNet | A | S-116, B36 |
| 6 | Plasticity (dormant neurons, srank) | no collapse, plateaus by 50k | A (negative) | B15 |
| 7 | Deeper search at test time (64 vs 16 sims) | lowered the score on 4 of 6 checkpoints; stuck-loop games run to the step cap | B | B24 |
| 8 | Deterministic evals | ~4 distinct games per checkpoint whatever the episode count; sticky p = 0.25 -> scores ~1/3 | A | B2 |
| 9 | LayerNorm everywhere; the PPO-era GTrXL agent | 0.0 at 10k; collapse root cause was the Kaiming init of the Impala encoder, not the gates | D | S-043, S-064 |
| 10 | CUDA-graphed search in training | bit-identical to eager at batch 32 and 256 (all weights, target, replay), 5.4-8.8x faster search | A | S-116, S-118 |

**Optimiser and gates.** The PPO-era note "gates frozen at init" (S-035-S-037) had a different root cause (S-043), so it says nothing
about the optimiser. In the EZ-V2 port the mixer's q/k/v/FFN get a tiny gradient (closed gate z ~0.13, small output projection) and SGD moves a
weight by lr x gradient. The CARLA head had the same family of problem and was fixed by exempting gates and norms from weight decay
(ch. 13.6c), so the lesson is shared across both domains: **gated residual blocks need a weight-decay exemption and an optimiser that is
invariant to gradient scale.**

## 4. External cross-check (2026-10-04)

- **Path + speed disentanglement.** CarLLaVA (Leaderboard 2.0 validation): waypoints only DS 3.21 and 0.68 static collisions; with a
  space-indexed path next to the time-indexed waypoints DS 4.49 and 0.0 static collisions. SimLingo reports the same effect (+39.9% DS,
  static collisions to zero). Our own audit finds the matching defect (row 3 above). This is TODO A15.
- **Data bucketing.** CarLLaVA trains on 650k samples per epoch from buckets: 5 acceleration/deceleration, 2 steering, 3 vehicle-hazard, stop
  signs, traffic lights, pedestrians, obstacle swerving, plus the full set. Its failure modes are rear-end collisions and high-speed
  merging (our families); temporal input gave fewer rear-end collisions qualitatively but no leaderboard gain; a rear camera helped lane changes.
- **LEAD / TFv6.** Aligning the expert with what the camera can see (ignore actors out of view, brake near visible hazards, cap speed by
  observable flow, larger boxes at turns): +1.37 DS on Bench2Drive (83.56 -> 84.94) and +11 on Longest6 v2. Removing the GRU decoder
  bottleneck +2.3, three target points +2.0. Camera-only 360 degrees 91.6, +LiDAR 94.7, +radar 94.2, all three 95.0.
  **The expert's data are public**: HF `ln2697/lead` (MIT, 269 GB, 8,930 routes, 43 scenario types, all 12 CARLA towns, six RGB cameras at 4 Hz,
  boxes/ego/route at 20 Hz in Arrow with a per-box `affects_ego` flag and occlusion counts, perturbed "recovery" views shifted 0.1-1.0 m and
  rotated 5-12.5 degrees), with camera-only checkpoints released (`ln2697/transfuser-carla-123d`).
- **Shortcuts.** PlanT 2.0 (2025) names exploitable shortcuts, rigid expert behaviour and overfitting to fixed expert trajectories as structural
  flaws of closed-loop driving models and argues for data-centric fixes.
- **Critics help.** Judge-Then-Drive (2026) refines a rough trajectory with a learned critic (73.3% Bench2Drive success, ~30% better in
  challenging scenarios); Hydra-MDP-style rule distillation is the same idea for camera-only inference (our A37).
- **Weight-space merging.** WiSE-FT (interpolate base and fine-tune) and model soups (average fine-tunes of one base) are established
  (Wortsman et al. 2022); we found no driving-policy application, which makes A36 cheap to try, not novel.
- **Search with learned models.** Epistemic MCTS (ICLR 2025) propagates epistemic uncertainty through the tree for deep exploration (AlphaZero,
  sparse reward); Model-Value Inconsistency (ICML 2022) builds an "implicit value ensemble" from one model and one value function, rolled out to
  different depths, as an uncertainty signal for pessimism; EZ-V2 itself trains Atari with SGD and proprioceptive control with Adam. We found no
  use of either signal in a Gumbel/EZ-V2 search.
- **Atari state of the art.** EZ-V2 (mean HNS 2.428) is still the reference on the 2026-10-04 search; our port matches it at 100k on one seed.

## 5. Corrections to earlier statements

- The 2026-10-04 TODO block first said recovery views and colour augmentation moved the score by +4 to +7 each. They did not reach significance
  (rows 6-7); it now says so.
- "Arm J removes E's overtaking deficit" (S-114) holds; "arm J improves the benchmark" does not (row 5): J is below E overall.
- B24 and B2: the 64-sim drop and the deterministic-eval limits are real, but part of the zero scores are stuck-loop games (no FIRE after a life
  loss) that run to the 27k-step cap; the eval JSON stores no episode lengths, so the share is unknown (B44).

Sources: [SimLingo](https://arxiv.org/abs/2503.09594), [CarLLaVA](https://arxiv.org/abs/2406.10165), [LEAD](https://arxiv.org/abs/2512.20563),
[LEAD dataset](https://huggingface.co/datasets/ln2697/lead), [LEAD data layout](https://github.com/autonomousvision/lead), [Safe2Drive](https://arxiv.org/abs/2606.00191),
[PlanT 2.0](https://arxiv.org/abs/2511.07292), [Judge, Then Drive](https://arxiv.org/abs/2604.27366), [Epistemic MCTS](https://arxiv.org/abs/2210.13455),
[Model-Value Inconsistency](https://arxiv.org/abs/2112.04153), [EfficientZero V2](https://arxiv.org/abs/2403.00564), [Model soups](https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf).

## 6. Addendum, 2026-10-04 evening (S-120): what the 220-route records add

Same records, three new scripts (`a3b_headroom.py`, `a3c_collision_anatomy.py`, the J-seed null pairs); numbers and caveats in `challenges/log_03` S-120.

| # | Finding | Evidence | Grade |
|---|---|---|---|
| 15 | **The frozen CARLA-pretrained backbone is robust to night, rain and fog; WoR is not** | E flat (night +0.4 vs day, rain -0.5 vs dry, fog +0.5 vs none); WoR fog -13.2 [-23.8, -2.9]; advantage over WoR in fog +22 vs +8 elsewhere (change +13.7 [+1.0, +26.8]; J +13.3 [+2.4, +24.9]); night and rain not significant | B (CI barely excludes 0, unpaired across conditions) |
| 16 | **J's vehicle collisions are systematic**: 80% of the points on routes where >= 3 of 4 seeds collide | 201 routes, 4 seeds; lead 89%, obstacle 83%, other 67% | A (descriptive; the seeds share one base) |
| 17 | **E and J drive straight into stopped vehicles** on obstacle routes; J also clips cones while passing | collision position projected on the route: E 0% off the line (0.06 m median), J vehicle 83% on the line, J layout 50% off the line to the left | A (descriptive) |
| 18 | **Repeated collisions cost J 3.5 points per route-run** (E 1.6, WoR 0.6) | 32% of J's colliding runs have >= 2 vehicle events | A (descriptive; the DS gain of a stop-after-contact behaviour is an upper bound) |
| 19 | **Obstacle routes are the universal failure** (E, J and WoR all < 50 on 29 of 196 routes, 21 obstacle) | per-route best-of-three | A (descriptive) |
| 20 | **E's lead-vehicle skill is generic, J's regression is a braking / speed regression** | E 82.2 on families it never trained on; J +20% speed, extra collisions on the line | B (mechanism inferred from speed and position; A36 step 0 and A45 test it) |
| 21 | **Closed-loop DS over 220 routes resolves ~5 DS (one run per arm)** | seed-pair null: SD 25.0 per route, spread +-1.9 of the mean | A (statistical) |

Camera-only reference (arXiv v1 table 8, one front camera): SimLingo DS 85.07, Emergency Brake 88.3 (ours 25-38), Overtaking 57.0 (16-18), Traffic Sign 82.5 (53-57), Give Way 53.3 (E 56), Merging 54.0 (E 44.9, J 26-39). The biggest gaps sit where our collisions are.
An open-loop vs closed-loop study (arXiv 2605.00066) reports that ego progress is the strongest single predictor of closed-loop success and that methods buying safety with slowness fall in closed loop (the WoR pattern here).
