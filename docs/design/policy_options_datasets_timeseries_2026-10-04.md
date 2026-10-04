# Policy-network options, elite CARLA datasets and a time-series view (2026-10-04, S-122 / S-123)

Why: the evidence of 2026-10-04 (S-117, S-120, S-121) says what is wrong with our policies; this note lists what else the field offers for the *network*, which data it has published
that fits **one front camera and our frozen CARLA-pretrained TF++ encoder**, and what the time-series-forecasting literature can and cannot add. Facts were checked against the
papers' pages or the dataset repositories on 2026-10-04 (links at the end); everything that is our own inference is marked as such. Items are TODO ids (`docs/todo/TODO_ACTIVE.md`).

## 0. Summary

- **CARLA network.** The best-evidenced change is still the output side: a **path head with a speed head conditioned on that path** (A15, upgraded to a cascade). Then the **input side**: our
  4x4 average pool of a stride-32 map is a very lossy tokenisation, and the closed-loop study BevAD found that *space-to-depth tokens plus token masking* moved Bench2Drive success 36.4% -> 57.4%
  (A54). A **generative head** adds little at our data size but keeps scaling when data grow (A55). A **ready-made counterfactual label source** exists (A56). Bigger heads, history/SSM trunks and
  time-series foundation models are not supported by the evidence.
- **Datasets.** The most usable new one is **SimLingo-Data**: same camera as our data (1024x512, FOV 110), PDM-Lite measurement/box layout we already load, 3.3M frames, plus per-frame
  **counterfactual crash labels (`dreamer/`)** and a **hazard-bucket index**. **Bench2Drive's and TaCarla's cameras differ** from the TF++ camera and cannot feed the frozen backbone as they are.
  LEAD stays the data-alignment option (A46). **Fail2Drive** adds 200 paired Town13 routes for evaluation (A58).
- **Time series.** What transfers: direct multi-horizon outputs (we already have them; the Atari world model does not), scale-free errors and a naive-forecast skill score (A53), the
  newsvendor view of asymmetric costs (a quantile that relaxes with waiting time, A57) and events / change points as auxiliary targets (B46). What does not: lookback windows (copy-cat risk;
  the strongest single-camera systems use one frame), foundation models (Chronos, TimesFM: evidence is aggregate traffic and crowd counts), SSM trunks.
- **Atari.** Make the mixer train first (B37). Then attack the compounding drift the way forecasters do: depth cap (B43), direct root-anchored heads (B45), an event / ball-forecast auxiliary (B46).
  Swapping to token, diffusion or actor-critic world models is not supported: on Breakout they sit at 16-302 against EZ-V2's 400.

## 1. What we run today

**CARLA (arms E and J).** Frozen CARLA-pretrained TF++ `regnety_032` (the TF++ release's backbone, 494 tensors) -> stride-32 map of the 288x768 frame (9x24) -> average-pooled to **4x4 = 16 vision
tokens**, + speed token (`Linear(1, d)`), route token (4 route points, `Linear(8, d)`), command embedding and a policy token -> transformer trunk (`qwen30m`: 8 layers, d = 512, 8 heads, 27.4M
parameters) -> `Linear` heads: **5 waypoints x 2 per command at 4 Hz (a 1.25 s horizon)**, L1 with lateral weight 3.0; an **8-bin two-hot target-speed head on the policy token** (loss weight 0.2,
decoded by the mean); PID control. The route is also painted into the image (overlay). 427k frames (6 towns + 20 obstacle archives for J), one front camera, recovery-camera augmentation on, no rail-Q
head, no history. Measured weaknesses (S-117, S-120): collisions are 80% systematic; E drives straight into stopped vehicles and J clips cones while passing; J regressed on lead-vehicle families
(braking / speed); the vision input is lightly used (no-vision MLP 0.7558 vs 0.5793); 220 routes resolve ~5 DS.

**Atari (EZ-V2 port).** `EZV2Model`: DownSample ResNet 96x96x12 -> **64x6x6 state**, conv dynamics with an action plane, value-prefix LSTM, 601-bin categorical value/reward heads (h-transform), policy
head, SimSiam consistency, Gumbel search. The thesis variant `trunk="gtrxl"` adds a **spatial** GTrXL token mixer (2 blocks, d = 128, over the 36 tokens of one state) after the representation and after
each dynamics step; it has **no temporal memory** (the 4-frame stack and the LSTM are the only time dimension). The mixer starts closed with a zero output projection; with SGD + weight decay its blocks did
not train (S-116, S-118, B36). Dynamics are unrolled 5 steps in training; a 64-simulation tree reaches depth 17 (S-116, S-121).

## 2. CARLA policy-network options

Ranking is by (evidence) x (fit to our measured failures) / (cost). "MDE" = what 220 routes can resolve (~5 DS at one run per arm, S-120): small effects are judged by mechanism metrics (A52).

| # | Option (TODO) | Evidence | Fit to our measurements | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | **Aligned path -> speed cascade**, horizon 1.25 -> 2.5 s (A15, A53) | BevAD: path + speed 57.4% SR vs waypoints 51.7%, static infractions 0.185 -> 0.055 (-70%). AlignDrive: speed as a 1-D displacement *along the predicted path* (conditioned on it), anchor-based, **89.07 DS / 73.18% SR**. CarLLaVA: DS 3.21 -> 4.49, static collisions 0.68 -> 0. SimLingo labels carry 10 waypoints | E / J hit stopped vehicles **on the route line** because the speed head is not tied to the path; J clips cones (lateral precision); our horizon is only 1.25 s | head-only on cached features, one arm | **first** |
| 2 | **Space-to-depth tokens + 20% token masking** (A54) | BevAD Table 1 (six cameras, Bench2Drive): baseline 36.4% SR / 66.9 DS; masking 42.0 / 72.4; + pixel-unshuffle p = 4: **57.4 / 82.6**; p = 5 collapses (40.9 / 66.4); unmasked attention sits on distant or occluded cells (causal confusion); the gain is invisible in open-loop L1 | our 16 tokens average a stride-32 map: a lead car at brake onset is 0.4-0.55 of one cell (S-100); `vision_grid 8` pooled the same map (ch. 13.24); vision is lightly used | head-only, cache 4x | Tier 2 |
| 3 | **Counterfactual critic with ready labels** (A56 -> A37) | Hydra-MDP; **VLAAD**: a collision-risk score added to TF++'s state, Town13 DS +14.1% relative, **86.97 DS / 71.97% SR on Bench2Drive** (arXiv 2603.25946). SimLingo's Dreamer files flag `dynamic_crash` for alternatives in every frame (60.5% of frames: accelerating would crash) | 80% of J's collision points are systematic, imitation never sees the unsafe speeds | label files ~9 GB, chunk ~25 GB, one GPU | Tier 2 |
| 4 | **Generative / multi-modal head** (flow matching or anchor MDN) (A55) | BevAD: diffusion 56.2% vs point 51.7% (trajectory); with P+S 59.4% vs 57.4%; dynamic infractions 0.423 vs 0.505; with cumulative data to ~16k scenes diffusion improves **linearly** while point estimators saturate after ~8k; BevAD-M 72.7% SR vs 55.3% for SimLingo data only | two-way obstacles need a committed stay / pass choice (A41); our heads are point estimators and data will grow 5-8x | head-only; head inference x steps | Tier 2 **only with scaled data** |
| 5 | **Newsvendor decode** (A57) | decision theory for asymmetric costs (time-series forecasting practice); A28's median decode -22.9 DS and S-059's hard brake both deadlocked; WoR pays 24 + 6 + 2 points for slowness | collision x0.6 vs the 200 s time cap | eval-only | Tier 1 with the matched-speed control |
| 6 | **Scale-free waypoint loss, residual to a constant-speed forecast** (A53) | forecasting practice: MASE, RevIN, skill against a naive forecast; for vehicles the constant-velocity forecast is the standard strong baseline | L1 in metres weights fast frames; hazard behaviour at low speed is small in metres | head-only | Tier 3, expect a small effect |
| 7 | Direct control head (TCP / Hydra-NeXt) (A22) | Hydra-NeXt 65.89 DS / 48.2% SR (arXiv abstract) | PID lag unproven (A11) | head-only | existing item |
| 8 | Scenario-expert MoE, bigger head (A23) | head size 10M-100M had no effect (ch. 13.24); Soft-MoE evidence is value-based Atari at 200M frames | none | - | skip |
| 9 | History tokens, recurrent / Mamba trunk (A48) | SimLingo (86.55 DS on the leaderboard) processes one image as two 448x448 tiles without frame stacking; CarLLaVA saw no DS gain from temporal input; DriveMamba-Tiny 53.5 DS | copy-cat shortcut (de Haan 2019, Wen 2020) | one arm | demoted behind 1-4 |
| 10 | Time-series foundation models (Chronos, TimesFM) | evidence is aggregate traffic speed and crowd counts, RL studies use classic control | none | - | skip |

**How the options map to our failures.** (a) on-line collisions with stopped vehicles: 1 (conditioning) and 3 (counterfactual); (b) cone clipping while passing: 1 (path precision) with A40;
(c) the lead-vehicle regression of J: 3 and the buckets of A56, data (A49 / A46); (d) repeated hits: A50; (e) the pass-or-stay bimodality: 4.

## 3. Elite CARLA datasets

First filter: **camera geometry**. Our frozen TF++ encoder was trained on 1024x512, horizontal FOV 110, camera at x = -1.5 m, z = 2.0 m (`src/config/camera.py`); a dataset with another front camera shifts every object
in the image and cannot feed the frozen backbone without re-rendering. Second: licence. Third: what extra labels it brings.

| Dataset | Expert | Size | Camera vs ours | Extras | Licence | Verdict |
|---|---|---|---|---|---|---|
| `autonomousvision/PDM_Lite_Carla_LB2` (have) | PDM-Lite | 5,134 routes, 38 scenarios, 8 towns, 581,662 frames at 4 Hz | **same** | boxes, route, `route_original`, `speed_reduced_by_obj_*` | HF card | baseline |
| **`RenzKa/simlingo` (SimLingo-Data)** | PDM-Lite (re-collected, random weather) | **3,308,315 frames, 1.17 TB** (driving data ~580 GB train: 1-scenario routes 274 GB, 3-scenario routes 202 GB, LB1-split 72 GB, parking Town12 31 GB; the ~580 GB validation split has the same format) | **same** (1024x512 front RGB; the SimLingo repo's `team_code/config.py` has camera_pos [-1.5, 0, 2.0], FOV 110) **+ an augmented-offset RGB per frame (recovery views)** | **Dreamer counterfactual labels (~9 GB)**, **bucket index (620 MB, 42 buckets, 2.92M frames)**, commentary, DriveLM VQA, LiDAR | Wayve non-commercial: academic research ok, attribution, models only published free / academic, **no use in operating a vehicle or robot**, revocable | **best fit** (A56) |
| `ln2697/lead` (LEAD / TFv6) | PDM-Lite variant aligned to the student's view | 8,930 routes, 43 scenario types, 12 towns, 73 h in the paper; ~1 TB all sensors, 269 GB in the HF listing I read | 360 rig, front geometry **not stated in the README**; the paper ablates a 110 deg front camera (check A46 step 0) | per-box `affects_ego`, perturbed views, camera-only checkpoints, Arrow / py123d | MIT | data-alignment option (A46) |
| `rethinklab/Bench2Drive` Base / Full | Think2Drive (RL world-model expert) | 1,000 clips ~335-400 GB / 13,638 clips ~4 TB, 10 Hz, 44 scenarios, all weathers | **no**: front camera 1600x900, FOV 70, x = 0.8, z = 1.6 | depth, semantics, boxes, 3D occupancy, expert value / features; per-scenario clip files | HF card Apache 2.0; README CC BY-NC-ND (check) | skip for the frozen backbone |
| `tugrul93/TaCarla` (+ labels) | PDM-Lite | 2.85M frames, 10 Hz, **Town12 + Town13 only**, 36 scenarios, 3.81 TB | **no**: nuScenes-style 6 cameras + LiDAR + 5 radars | rarity scores for long-tail mining | paper: arXiv licence; check the dataset card | skip (cameras) |
| `rethinklab/Bench2Drive-Speed` | Bench2Drive-style | 2,100 scenarios, 218 GB | **no** | desired-speed and overtake / follow commands (a speed-conditioned policy) | CC BY-NC-ND | idea only: a trained caution knob (A38 step 3) |
| CARLA-Collide (VLAAD) | TF++ failures | 1,521 collision clips (920 vehicle, 26 pedestrian, 98 layout in train), Town12 / 13, RGB at 4 Hz + control | n/a | collision captions | **no release found** | generate our own (A44) |
| `autonomousvision/fail2drive` (evaluation) | PDM-Lite, TF++ agents | 100 route pairs = 200 routes, Town13, 17 unseen long-tail classes | n/a (simulator) | the pair isolates the shift; SOTA models drop 22.8% on average | MIT | **extra routes** (A58) |

**SimLingo-Data details I verified by opening the files (`scripts/analysis/simlingo_dreamer_probe.py`, `E:\MThesis_EXP\analysis_20261004\simlingo_dreamer_probe_0410.txt`).** One Dreamer archive
(LB1-split ControlLoss, 23,091 frames): per frame the file holds counterfactual alternatives with their own waypoints / route and a kinematic roll-out verdict. Share of entries with `dynamic_crash`:
`faster` (throttle ramp to ~14.8 m/s) **60.5%**, `stop` 46.6%, random `target_speed` 43.2%, `faster_factor` 10.7%, `slower` 4.3%, `slower_factor` 0.2%, `crash` (steer at a named vehicle: id,
distance, type) 96.2%; `lane_change` entries carry `allowed` (sidewalk / oncoming lane: 41% not allowed). The bucket pickle (no import opcodes, loaded without globals) holds the CarLLaVA / SimLingo
sampling buckets: `brake` 1.49M, `leading_object_vehicle` 1.92M, `vehicle` 630k, `vehicle_side` 382k, `vehicle_front` 41k, `red_light` 271k, `stop_sign` 172k, `changed_route` 177k (swerves),
`walker_hazard` 26k, `leading_object_static.prop.trafficwarning` 36k (obstacle scenes), `start_from_stop` 97k, `acceleration_-5` 127k, `target_speed_*`, `speed_limit_*`, `lateral_control_*`.
Caveats: the data are a separate collection from `PDM_Lite_Carla_LB2` (the README says the 3.3M frames are not from unique routes); chunk files mix scenario types (about 23 GB each), so a hazard subset
still costs whole chunks; Dreamer's crash test is open-loop against recorded actor futures, the same assumption as A37.

**What each dataset would fix** (our measurements): lead-vehicle regression -> SimLingo-Data buckets (`vehicle_front`, `leading_object_vehicle`) and Dreamer `faster` / `stop` crash labels; obstacle
universal failures -> LEAD's aligned expert (visibility asymmetry on two-way passes, A21) and SimLingo's 1-scenario obstacle routes; the evaluation floor -> Fail2Drive. Sizes: scaling data 5-8x only
helps a point-estimator head up to ~8k scenes (BevAD), which is why A55 comes with the data upgrade.

## 4. A time-series view

Our CARLA policy *is* a multi-horizon forecaster of the ego's own future (5 waypoints, a speed distribution) conditioned on exogenous covariates (image, route, command); our Atari world model is a
multi-step forecaster of latent states. The forecasting literature has opinions about both.

**What transfers.**
1. *Direct multi-horizon beats iterated forecasting.* Zeng et al. (DLinear, AAAI 2023): the long-horizon accuracy of Transformer forecasters comes mostly from non-autoregressive direct multi-step output, not from
   attention; a linear model with trend / remainder decomposition beat them on most datasets. CARLA: we already predict directly (nothing to change; the path / speed split is a decomposition). **Atari: our
   dynamics model iterates, and the drift at depth 6-12 is 2-4x worse than inside the trained depth (S-121) -> B45, direct root-anchored heads.**
2. *Scale-free errors and a naive baseline.* Forecasters are trained and scored on errors divided by a naive forecast's error (MASE) or on instance-normalised series (RevIN), so large-scale series do not dominate, and
   a model is judged by its skill over the naive forecast (constant velocity for vehicles). Our L1 is in metres, so fast frames weigh more; hazard behaviour (starting, stopping) lives at low speed. -> A53 (an honest note:
   L1 gradients are sign x weight per element, so the 73.5% "longitudinal share" of the loss *value* (ch. 13.26) is not a gradient share; the argument is about scale across frames, not about budget across axes).
3. *Asymmetric costs -> a quantile, not the mean* (the newsvendor result): the optimal point forecast for cost c_o per unit above and c_u per unit below is the quantile q = c_u / (c_o + c_u). A collision multiplies the
   score by 0.6, waiting costs little until the 200 s cap, so the right speed is a *low* quantile of the posterior that **rises as the car has waited** (the cost of waiting accumulates). A static low quantile deadlocked (A28,
   S-059); a quantile with a waiting-time schedule is the principled stall breaker -> A57, tested against a global speed-scale control.
4. *Events and change points as auxiliary targets.* The Event-Aware World Model (arXiv 2601.19336) predicts automatically extracted events and reports 10-45% over MBRL baselines on Atari 100K, Craftax and DMC. For Breakout
   the events are few and exact (wall bounce, paddle hit, brick hit, ball lost) and the ball path is piecewise linear -> B46. For CARLA the analogue is brake onset (A38).

**What does not transfer.**
- *Long lookback.* Forecasters gain from history because the target is the continuation of the same series; for imitation the ego's own past speed and actions leak the answer (copy-cat, inertia). The strongest single-camera system
  on the leaderboard (SimLingo) uses one image; our A48 stays behind the cheaper items.
- *Foundation models (Chronos, TimesFM, Moirai).* Published evidence is aggregate: traffic speed and crowd counts (a transportation benchmark, arXiv 2602.24238; a pedestrian-count study), and an RL repository that uses Chronos / Moirai as
  dynamics models of classic-control tasks and as energy forecasters (tsfm-rl). Nothing shows a benefit for image-conditioned control or for event-driven braking.
- *SSM / Mamba trunks.* DRAMA (Mamba-2 world model, 7M parameters) is competitive on Atari 100k and DriveMamba-Tiny reaches 53.5 DS on Bench2Drive; their advantage is long sequences, our unroll is 5 and our CARLA input is one frame.
- *Probabilistic foundation forecasters for the evaluation noise.* The collision lottery is mostly systematic (S-120), not a stochastic process to forecast.

## 5. Atari policy-network options

Breakout references (100k steps, per-game tables): **EZ-V2 400.1** (paper; our port 363.1 at 50k, 321.2 at 100k), Delta-IRIS 302, DIAMOND 133, EMERALD 62, TWISTER 35, DreamerV3 31, STORM 16 (EMERALD, arXiv 2507.04075, Table 11 and the
TWISTER paper). Search with a value-equivalent model dominates this game, so the thesis comparison stays inside EZ-V2's family.

| # | Option (TODO) | Evidence | Fit | Verdict |
|---|---|---|---|---|
| 1 | **Make the mixer train**: AdamW parameter group without weight decay, non-zero output init, open gates, module-update audit (B37) | weight audits: only the two mixers are dead (S-118); wd 0 alone left blocks at init | blocks the whole "GTrXL vs ResNet" question | **first** |
| 2 | **Depth cap, then direct root-anchored heads** (B43 -> B45) | B43 on nine checkpoints: policy KL and value error x2-4 at depth 6-12 (S-121); DMS vs iterated forecasting (DLinear study) | the search leaves the trained regime (tree depth 17 vs unroll 5) | Tier 1-2 |
| 3 | **Event / ball-forecast auxiliary** (B46); action-conditioned CPC (B11) | EAWM +10-45% on MBRL baselines; TWISTER's AC-CPC (162% mean, 77% median HNS without search) | Breakout dynamics are a ball path with reflections | Tier 3 |
| 4 | **Pessimistic search from two-network disagreement** (B38) | EMCTS and the model-value inconsistency papers | imagined values drift low at depth, variance and the winner's curse | Tier 1 |
| 5 | Temporal memory in the latent (UniZero-style latent history, B12) | UniZero: single-frame inputs match 4-frame MuZero, better on memory tasks | our "XL" is spatial only | Later |
| 6 | Trunk swaps (ViT / ConvNeXt / Soft-MoE) | Soft-MoE: Rainbow +20% with 8 experts at 200M frames, plain widening -40% | only after the mixer question is settled; 100k is not 200M | Later |
| 7 | Token / diffusion / actor-critic world models, Mamba dynamics | Breakout 16-302 (above); DRAMA competitive overall with 7M parameters | search is what wins here; our unroll is 5 | not recommended |

## 6. Recommended order (MDE-aware)

1. **No box:** A58 wiring plan, A56 pilot design; run the Dreamer probe on more archives; A44 / A50 / B37 code (already planned).
2. **Data box (150-300 GB, one GPU):** A56 pilot (one SimLingo-Data chunk + its Dreamer chunk, stream-extract RGB + measurements + boxes, cache features, critic AUROC by stratum), then A15 with the aligned cascade (A53's horizon
   and loss as options), A54 tokens, each one change per arm and judged by mechanism metrics (collision events per run by group, off-line share, time-cap share) next to DS (A52).
3. **Eval boxes:** A44 + A50 + A57 + the speed-scale control on J; A36 merges; A58 Fail2Drive pairs for E, J, WoR.
4. **Atari box:** B37 probe -> B36 rerun; B43 step 1 and B45 after.
5. A55 only together with 3x data (A56 / A46).

## Sources

SimLingo [arXiv 2503.09594](https://arxiv.org/abs/2503.09594) and dataset [RenzKa/simlingo](https://huggingface.co/datasets/RenzKa/simlingo) (licence file read in full); BevAD
[arXiv 2603.15185](https://arxiv.org/abs/2603.15185); AlignDrive [arXiv 2601.01762](https://arxiv.org/abs/2601.01762); VLAAD / CARLA-Collide [arXiv 2603.25946](https://arxiv.org/abs/2603.25946); RoG-DAgger
[arXiv 2608.24525](https://arxiv.org/abs/2608.24525); LEAD [arXiv 2512.20563](https://arxiv.org/abs/2512.20563) and [repo](https://github.com/kesai-labs/lead); Bench2Drive
[repo](https://github.com/Thinklab-SJTU/Bench2Drive) and [data](https://huggingface.co/datasets/rethinklab/Bench2Drive); Bench2Drive-Speed [data](https://huggingface.co/datasets/rethinklab/Bench2Drive-Speed); TaCarla
[arXiv 2602.23499](https://arxiv.org/abs/2602.23499) and [data](https://huggingface.co/datasets/tugrul93/TaCarla); Fail2Drive [arXiv 2604.08535](https://arxiv.org/abs/2604.08535) and
[repo](https://github.com/autonomousvision/fail2drive); open-loop vs closed-loop metrics [arXiv 2605.00066](https://arxiv.org/abs/2605.00066); Hydra-NeXt [arXiv 2503.12030](https://arxiv.org/abs/2503.12030); DLinear
[arXiv 2205.13504](https://arxiv.org/abs/2205.13504); time-series foundation models in transportation [arXiv 2602.24238](https://arxiv.org/abs/2602.24238) and [tsfm-rl](https://github.com/JohannesWittmann9/tsfm-rl); DriveMamba
[arXiv 2602.13301](https://arxiv.org/abs/2602.13301); EMERALD [arXiv 2507.04075](https://arxiv.org/abs/2507.04075); TWISTER [arXiv 2503.04416](https://arxiv.org/abs/2503.04416); DRAMA
[arXiv 2410.08893](https://arxiv.org/abs/2410.08893); EAWM [arXiv 2601.19336](https://arxiv.org/abs/2601.19336); UniZero [arXiv 2406.10667](https://arxiv.org/abs/2406.10667); Soft-MoE in RL
[arXiv 2402.08609](https://arxiv.org/abs/2402.08609); EfficientZero V2 [arXiv 2403.00564](https://arxiv.org/abs/2403.00564).
