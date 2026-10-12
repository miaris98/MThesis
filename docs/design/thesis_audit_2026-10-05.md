# Thesis_typist audit — claims, facts and references (2026-10-05)

Scope: everything compiled into `Thesis_typist/MIT-Thesis.typ` (front matter, chapters 1-7, Appendix A listings and the
challenges summary, `bib/references.bib`), plus `figures/generate_figures.py` and `presentation/slides.typ`.
`Thesis_typist/challenges/*.typ` are byte-for-byte copies of `challenges/*.md` (4-line header diff), so they are the project's own
logs, not separate claims; they were used as evidence, not audited line by line.

Method. Every quantitative or factual claim was checked against, in this order: the code (`src/`, `atari_qwen/`, vendored
`Carla-utils/` configs), the project logs (`challenges/`, `docs/design/what_has_worked_2026-10-04.md`,
`docs/design/continuation_prompt.md`, TODO files, `.agents/rules`), the external archive (`E:\MThesis_EXP`), and external sources
(arXiv API, Crossref, the NTUA DSpace repository, the official Bench2Drive route file, and the SPR / DER / EfficientZero / EZ-V2 / BBF
papers). The thesis compiles cleanly with Typst 0.15.1 (114 pages, no warnings) and prints all 130 bib entries (`full: true`).

Labels. **CONTRADICTED** = a record, code, or external source says otherwise. **NO SOURCE** = I found no run, log, or file that
produces the number (the claim may still come from something I did not find; point me at it and I will re-check).
**WRONG DETAIL** = real thing, wrong parameter or attribution. **MINOR** = wording, typography, consistency.

What I could not do: re-run any experiment; read `.env` (so the Hugging Face repos cannot be checked); query DBLP (it started
serving a bot-challenge page, so venue/page numbers for some classic papers rest on arXiv/Crossref/my own knowledge, which I mark
where it matters).

---

## 1. Bottom line

1. **Chapters 3-6 do not describe the system and results that the repository actually contains.** Method details (RoPE,
   autoregression, layer count, token sequence, controller, camera pose, speed bins, losses, hyper-parameters) differ from the code,
   and most headline result tables (5.1-5.14, 6.x) have no matching run; several are contradicted by your own logs.
2. **Three of the four figures are generated from synthetic or hard-coded numbers** (`figures/generate_figures.py`), including the
   "Validation ADE learning curves", which the script itself labels "Simulated learning curves".
3. **The strongest real result is missing.** Arm E beats the original World-on-Rails on all 220 Bench2Drive routes
   (official 63.4 vs 49.4 DS, paired +13.7 [+8.6, +18.8], log S-117 of 2026-10-04; unseen towns +15.9, train towns +1.3 tie). The thesis instead reports a
   19-route number (65.4) beside a WoR number (66.9) that is *higher*, while the text and slides say "outperforms WoR".
4. **Two claims the logs say are not established are presented as findings**: geometry loss (closed-loop gain 0, sign flipped twice)
   and recovery-camera augmentation (+3.1 [-1.8, +9.5] after repeats, grade C).
5. **The Atari chapter has no multi-game results behind it.** The only archived Pong 100k run scores -21.00 at every eval
   (the thesis says 12.4, "human-level by 60k"). The "zero-copy shared-memory replay engine, 18 → 74 SPS" does not exist in the code.
6. **The SLR numbers do not add up** (642 → 488 → 194 → 78 excluded = 116, text says 126) and there is no search log anywhere in the
   repo; the bibliography itself violates the stated inclusion criteria.
7. **References: 14 of the 88 cited entries have a wrong title/author/venue/identifier**, one cited paper does not exist under the
   given ID, and 42 uncited entries are printed in the reference list anyway.
8. What does hold up is listed in §8; it is a real and sizeable core (Section 8 + the challenges logs).

Recommended order of repair is in §9.

---

## 2. Results chapter (Ch. 5) — traceability

### 2.1 Table 5.1 (`05_empirical_results.typ:21-41`, caption says "220 routes")

| Row | Thesis | What I found |
|---|---|---|
| PDM-Lite | 96.40 / 100% / 0.96 | Published Bench2Drive table lists 97.02 DS / 92.27% SR (`docs/design/eval_tiers_design.md:376`). |
| TF++ | 80.33 / 96.5% / 0.83 | The author's own TF++ run on the **19-route** b2d20 subset (ledger: "TF++ 80.3 on our 19 routes"); archived JSON `E:\MThesis_EXP\bench2drive_eval\run_20260916\bench2drive_out\b2d20_tfpp_v2.json` gives 81.02 on 16 routes. Published TF++ = 84.21 DS / 67.27% SR. Not 220 routes. |
| InterFuser | 76.21 | Not in the Bench2Drive published table. 76.18 is its Leaderboard-1.0 DS (a different benchmark). |
| TCP | 61.80 | Bench2Drive published: **40.70 DS / 15.00% SR** (`eval_tiers_design.md:369`). The paper's own LB1.0 number is 69.7, Town05-Long 57.2. 61.80 matches none. |
| World-on-Rails | 66.90 | Mean of two runs of the *released* WoR agent on the **19** b2d20 routes (67.8 and 66.0; memory `wor-original-vs-arms`). On all 220 routes WoR is ≈45.6-49.4 (official). |
| CILRS | 38.25 | No source. |
| Conv baseline | 59.14 | The **pre-2026-09-13** number, measured through the wrong camera (x +1.3, z 1.3, FOV 100, 256×256) and a constant 20 km/h controller (target-speed head output discarded). `continuation_prompt.md:68`: "the old 59.14 was measured through the broken camera"; memory `closed-loop-constant-speed...`: "not comparable to any run made after the fix". |
| Qwen-30M nominal | 56.86 | Same as above. The 2026-09-16 re-runs (also invalid, S-022 agent-config mutation bug) give DS 21.1 (cnn) and 22.3 (qwen) on 16 routes in `...\run_20260916\bench2drive_out\b2d20_{cnn,qwen}.json`. |
| Qwen + geometry | 61.42 | No run. Ledger row 10 / ch. 13.33: geometry loss, "closed-loop gain 0 at 3 seeds (sign flipped twice)". |
| **Qwen + recovery (Ours)** | 65.40 / 91.2% / 0.78 | 65.4 = **arm E e15 on 19 routes** (E = 288×768 input + recovery camera + colour-aug family). Recovery camera alone (A−B) = +3.1 [-1.8, +9.5], grade C. |

Consequences: the "+27.15 over CILRS / +3.60 over TCP" sentence (`:50-51`) compares a 19-route number with a 220-route number or
with unsourced values; "closing the gap with multimodal fusion" (`:52-54`) compares RC of one model with DS of another; and in this
very table WoR (66.90) is above "Ours" (65.40), yet `slides.typ:291` says "outperforming ... World-on-Rails (66.9 DS)".

### 2.2 Head comparison (`:59-118`, Table 5.2, "13.6%")

* **The 13.6% is a reduction in held-out *loss*, not ADE.** Real (`challenges_13` §13.20/13.23, seed 0, 4 towns = Town01+02+03+10,
  656 routes, 114,556 frames): `qwen30m` 0.5715 vs `cnn` 0.6616 on `longitudinal + 3×lateral`, paired diff -0.0900,
  CI [-0.1160, -0.0670], 78/98 routes. (0.6616-0.5715)/0.6616 = 13.62%. Seeds 1/2 give -0.0757 and -0.0718, i.e. **10.8-13.6%**; grid 8 gives 11.8%.
  The thesis re-labels it "waypoint accuracy" / "ADE 0.384 m → 0.331 m" (13.8%) in Ch. 5, "0.385 → 0.333" in the table means
  (13.5%) and "0.381 → 0.329" on the slides (13.65%); none of these metre values appears in any log.
* **"Held-out towns" is wrong.** All six runs use the same 98 held-out *routes* of the *same four towns* (`--split_seed 0`, §13.23).
  The one held-out-town check (Town04, 10 routes, §13.32) came back NOT SUPPORTED and cnn nominally ahead.
* **The small-data numbers are mis-assigned.** The real 5.45× gap is Town01-only (129 routes, 9,632 train frames): train 0.1339 vs
  held-out 0.7298; the 4-town gap is 1.52×. The thesis says "Town01 & Town02, ~30k frames, L_val = 0.272, L_train < 0.05". The paired
  result on Town01 was **not significant** (§13.17), and there is no closed-loop RC comparison "82.3% vs 82.7%" for that regime.
* **Table 5.2 is internally inconsistent**: conv RC is 82.3% in both the small- and large-data regimes; the conv ADE 0.384 is reused for the
  small regime; Table 5.8's "Pure L1" ADE (0.384) equals the conv head and "Full geometry" (0.331) equals the "Qwen trunk"; and in Table 5.1
  the Qwen nominal model is *below* the conv head (56.86 vs 59.14 DS, 79.1% vs 82.3% RC) while the text claims the transformer "separates decisively" with 89.4% RC (no such row).
* **Statistics (`:105-113`) cannot be reproduced** from the thesis's own numbers: Welch on the Table 5.2 ADE seeds gives t = 14.48, df = 3.9,
  p = 1.5e-4 (thesis: t = 12.48, p = 3.2e-5); a Mann-Whitney U = 0 has exact two-sided p = 0.10 for 3 vs 3, 0.0079 for 5 vs 5, 1.3e-8 for 15 vs 15 (thesis: 0.0039);
  a DS CI of [63.8, 67.1] contradicts S-120 ("220 routes resolve ~5 DS at one run per arm").
* **What the logs do support**: transformer head > WoR conv head in closed loop, controlled (arm H 50.4 vs E 65.4, -15.0 [-25.4, -6.0], grade A),
  and the older 3-seed Town01 closed-loop result (§13.33). Neither is in the thesis in that form.

### 2.3 Tables with no experiment behind them

| Thesis item | Evidence it was not run / is contradicted |
|---|---|
| Table 5.3 per-town DS/RC/speed (8 towns) | Bench2Drive-220 has **12 towns** (Town12 104, Town13 47, Town04 12, Town03 11, Town05 9, Town11 7, Town15 7, Town06 6, Town07 5, Town10HD 4, Town02 4, Town01 4; official XML). 151/220 routes are Town12/13. The thesis gives Town12 = 30 routes and the 40/35/65/50 grouping. I found no per-town DS table matching these values (the logs only give seen-vs-unseen-town groups). |
| Table 5.4 scenario clusters (42/35/45/48/50 routes) | Rows duplicate the Town03 and Town07 numbers; route-weighted DS = 62.11, not 65.40. Not Bench2Drive's categories. |
| Table 5.5 "six clusters of 44 scenarios, 84-96% success" | The 44 scenarios are named (Accident, ParkedObstacle, HardBreakRoute, ...), 5 routes each, 23 weather settings; "Adverse Weather Transitions" is not a scenario. Success rates of 84-96% are incompatible with DS 65 (published TF++ SR is 67%). |
| Case studies 1-3 (Town03 "scenario 19", Town04 "11", Town07 "35") | No telemetry for these routes. They rely on a "[0.0, 0.2] m/s creep bin" that does not exist (bin 0 is exactly 0.0) and a 20 m/s cruise target that contradicts Ch. 3's [0, 10] m/s range (§3, M8). |
| Table 5.7 speed head: continuous L1 / Smooth-L1 / 8-bin | TODO C24 "Done" (head on in every arm), C35 Smooth-L1 "Deprioritised"; no ablation run. The code decodes the *mean* of the bins, which its own docstring says brings creep back (`aux_heads.py:112`); median decode was tested: -22.9 [-33.5, -12.6] (S-112). |
| Table 5.9 coordinates ("within 4 cm at k=5") | Val ADE of the same model is 0.33 m; x/y convention is reversed vs Ch. 3. |
| Table 5.10 recovery offset ±0.2/±0.4/±0.6 m | TODO C3 "Tried, not shown": A 60.8 vs B 53.6 (S-073), gap shrank to +3.1 [-1.8, +9.5] with repeats (S-079). The offsets in the data are **random per route** (DataAgent), not a swept parameter. |
| Table 5.11 horizon K ∈ {3,5,8,10} | TODO C30 is **open** (not run). ADE across different K is also not comparable. |
| Table 5.12 temporal window T ∈ {1,2,4,8}, "4-frame history (Ours)" | TODO C14 is **open**. The real model is single-frame (also contradicts Ch. 3's one-frame architecture and Case Study 1's "temporal token persistence"). |
| Table 5.13 weather (monotone 68 → 58) | TODO C4/C50 "Deprioritised". S-120/ledger #15: the frozen backbone is **flat** across night/rain/fog (+0.4/-0.5/+0.5); "Dense Fog" is not one of CARLA's 14 presets that Ch. 4 says were used (Bench2Drive's own 23 weather settings do vary fog density, but the thesis does not say it used them). |
| Table 5.14 RTX 4090/3090/3080 Ti/T4/**Jetson AGX Orin/TensorRT INT8** | TODO C84 (TensorRT INT8), C85, C87 "Resolved - Out of scope". No profiling script. 3080 Ti slower than 3090 by 75% is implausible; Ch. 2 Table 2.3 gives the same model 14 ms / 2.4 GB on the 3090 vs 4.8 ms / 2.18 GB here; slides: 9.8 ms. |
| Table 5.6 infractions per km (incl. TF++ row), "0.00 pedestrian collisions for all learned policies" | No per-km table in the logs; the ledger says the opposite for pedestrians: for arm J, DynamicObjectCrossing runs collide 95% of the time. |
| "Perfect/within 4 cm/84.1% RC in dense fog" etc. | follows from the above. |

### 2.4 Chapter 1 / abstract headline claims

* Intro `:202-203` "improving Bench2Drive driving scores by **87%**" vs Ch. 2 `:324` and Ch. 5 "**out-of-lane infractions -87%** (1.42 → 0.18/km)".
  The table gives DS 56.86 → 65.40 = +15%. The two statements cannot both be right; the 87% is the infraction figure.
* Intro `:197-200` and abstract: "multi-seed replications across held-out towns ... +13.6% waypoint accuracy" — see §2.2.
* Conclusions RQ2: "geometry loss reduces lateral jitter 50%" — the real effect is on *heading error* (-92%: 0.043 → 0.0034) with val lateral error flat
  (0.0517 vs 0.0522) and no closed-loop gain (§13.29, §13.33).
* Ch. 2 Gap 2 "matches the closed-loop survival of multimodal pipelines" — contradicted by Table 5.1 itself (65.4 vs 80.33).

---

## 3. Methods (Ch. 3-4, Appendix A) vs the code

| # | Thesis | Code / record | Where |
|---|---|---|---|
| M1 | Rotary position embeddings in every block; Algorithm 3.1 `ApplyRoPE`; Table 3.1 "RoPE" | **No RoPE anywhere** (`grep -i rope` and `grep -i rotary` over all `.py` files match nothing but the word "property"). Attention is plain SDPA (`layers.py:55-74`); position info = learned `vision_pos` + `type_embed` (`qwen_wor_policy.py:74-75`). | Ch. 1, 2, 3 |
| M2 | "Autoregressive" Qwen head (≈25 places) | No causal mask; one-shot MLP head over the policy token; TODO C86: "policy already decodes all waypoints in one pass (parallel head)". | all |
| M3 | 6 layers, d_ffn = 682 | `qwen30m` = **8 layers, ffn_dim 1536** (`qwen_wor_policy.py:39`); trainable trunk 27.5M (§13.25). | Ch. 3, 4 |
| M4 | 96 visual tokens + 3 = 99-token sequence | Sequence = `[policy, vision cells, speed, route, command]`; arms A-I use `vision_grid 4` → **16 vision tokens, 20 total** (`TODO_PDM_LITE...:36` "16 vision + 4 state"). Slides say 49 tokens. | Ch. 3 Table 3.1, Ch. 4 |
| M5 | Navigation by a command token {Left, Right, ...}; no route input | PDM-Lite's command enum is **LANEFOLLOW on every frame** (challenges_09 §9.5-9.6; `train_wor.py --route_points` help text); the **route points (+ painted route overlay)** are the only navigation input and the ledger's #1 confirmed lever. Absent from Ch. 3. | Ch. 3 |
| M6 | RegNetY-32: stage depths 2/4/14/2, widths 32/80/224/512, stage-4 512 ch, `Linear(512,512)+LayerNorm` | `regnety_032` (RegNetY-3.2GF) from the TF++ release; stage-4 output is **1512×6×16**; code is `nn.Linear(1512→512)` with no LayerNorm. Table 3.1 stride arithmetic is also inconsistent (stage 4 "stride 2" cannot go 24×64 → 6×16). | Ch. 3 |
| M7 | Camera at (x 1.3, y 0, z 1.8), pitch -5°; "matches TF++ geometry"; "discard bottom 25% and the sky; c'_y = 256 - 160" | Data/TF++ camera = **(x -1.5, y 0, z 2.0), pitch 0, FOV 110, 1024×512** (`camera.py:39-49`, `config.py:355-361`). 1.3/1.3/FOV100 was the old *eval* camera. Crop is **bottom 25% only** (512 → 384) then resize to 512×192 (so principal point does not move); arms A-I were trained with `crop_bottom_frac 0.0` (S-103). | Ch. 3, 4 |
| M8 | Target speed: 8 bins over [0, 10] m/s, zero bin [0.0, 0.2]; MLP 512→128→8 | TF++ bins **(0, 4, 8, 10, 13.89, 16, 17.78, 20) m/s**, exact-zero first bin, MLP →256→8, mean decode (`aux_heads.py:36,84-120`). The thesis's own Case Study 2 uses 20 m/s. | Ch. 3, 5 |
| M9 | Adaptive Pure Pursuit (lookahead 0.8 v + 2.5, wheelbase 2.8) + "Stanley"; PID Kp 0.50; cites TransFuser for pure pursuit | `PIDController`: **PID on heading angle to the 3rd waypoint** (kp 0.75, ki 0.05, kd 0.15); speed PID **kp 1.0**, ki 0.05, kd 0.10 (`pid_controller.py`). Also `clip(angle in rad, ±1)` would be a unit error against CARLA's normalised steer. | Ch. 3, 5 |
| M10 | Loss: flat L1 + 0.5·heading(atan2 diff, 1-cos) + **0.1**·curvature(κ formula) + 0.2·speed; "grid search establishes optimal weights" | `longitudinal + 3.0 × lateral` L1; heading = 1-cos on **unit segment vectors** (atan2 explicitly rejected), masked < 0.10 m, ego origin prepended; curvature = L1 of **second differences**; weights run: **0.5 / 0.2** (§13.29). No grid search found (ledger: speed-loss weight "never swept"). Appendix A code uses yet another heading definition (mean absolute wrapped Δθ). | Ch. 3, 4, 5, App. A |
| M11 | "73.5% of the backpropagated gradient" `∂L/∂y ÷ (∂L/∂x+∂L/∂y) > 0.735` | 73.5% is the **loss-value share** (0.43337 + 3·0.05206 = 0.58955; 0.43337/0.58955). For L1 each coordinate's gradient is ±1/K (±3/K with the 3× weight), so the gradient share would be 50% / 25%, not 73.5%. (`wor_eval.py:40` and §13.26 also say "gradient", but the number is a loss share.) | Ch. 3, 7, slides |
| M12 | Alg. 3.2: `x' = (x - Δy)·exp(-k/τ)`, τ = 2.5, "converges within 1.5 s"; Ch. 2: SE(2) ±15°; Ch. 4: fixed cameras at ∓0.4 m / ±15° mounted by the author | Real: the **public PDM-Lite DataAgent** renders one augmented camera per frame at a **random per-route lateral + yaw offset** (`augmentation_translation/rotation`), partially covered; `wor_dataset.py:28-60,347-375` re-projects targets with a rigid transform, **no decay**. `exp(-1.2) = 0.30` at 1.5 s, so "converges within 1.5 s" is false; and `exp(-k/τ) → 0` converges to the *camera heading line* (parallel to the lane), not back to the centreline. Left/right sign conventions differ across Ch. 2/3/4; Ch. 2 says "rotated inward" while Table 4.1 gives yaw -15° for the left camera (outward in CARLA). | Ch. 1-4 |
| M13 | Data "collected using PDM-Lite"; 100 / 214 / 656 / 1,312 / 220 routes (~15k/30k/250k/500k/95k frames); 4-town = Towns 1,3,4,6; held-out Towns 5,7,12 | The **public** `autonomousvision/PDM_Lite_Carla_LB2` set was downloaded (challenges_09 §9.1; `scripts/setup/download_pdm_lite.py`); arms A-I = Town01-05+10; the 4-town set of §13.20 = Town01+02+03+10, 656 routes, **114,556 frames** (the ~250k figure belongs to the 8-town set: 1,730 route groups, 250,945 frames); Town01-only 129 routes / 22,376 frames; 2-town 214 routes / 35,136 frames. Evaluation is closed-loop on Bench2Drive, not a held-out frame split. | Ch. 4 |
| M14 | Training: AdamW lr 2e-4, wd 1e-2, batch 64, FP16 + GradScaler | AdamW **lr 3e-4**, 5% warmup + cosine, **batch 32**, grad-clip 5.0, gates/norms in a no-decay group; **bfloat16** since §13.34 (fp16 overflowed the residual stream). | Ch. 4 |
| M15 | "Conv head 1.2M params", "30M Qwen" | cnn head 1.8M; Qwen trunk 27.5M trainable (~30.8M with projections). | Ch. 7 |
| M16 | `alpha` "learnable scalar, near-zero (0.1)" / Appendix A `torch.ones(1)` "initialised to 1.0" | `0.1 * torch.ones(dim)` (per-channel vector). Ch. 3 and App. A disagree with each other and with the code. | Ch. 3, App. A |
| M17 | Atari model: 4-layer Qwen block, d = 256, 121 tokens, Impala 64×11×11, AdamW | The champion path is the EZ-V2 port: ResNet trunk + **GTrXL token mixer over 36 spatial tokens (2 blocks, d = 128, closed gates, no memory)**, SGD (EZ-V2 trains Atari with SGD), Gumbel search with 16 simulations by default (B24 compared 64 vs 16); the PPO-era Impala encoder is 16/32/32 channels (not 64). The mixer itself was dead (S-116). The Qwen actor-critic is the earlier PPO-era model. | Ch. 3, 4, 6 |
| M18 | `multiprocessing.shared_memory` zero-copy replay engine; shared-memory sensor rings; "18 → 74 SPS, 4.11×" | `grep shared_memory` finds only `shared_memory=False` in a LightZero config. Atari envs = Gymnasium `AsyncVectorEnv`; replay is a numpy 2-D ring (S-053). CARLA vector env = N shared servers + `--sim-dt` (challenges_04 §4.5-4.6). No log contains "74 SPS" or "4.11". **Measured Atari rate: 6.1-7.0 min / 1k steps (≈2.5 SPS), 30k = 3.3-3.5 h** (`log_06:554`). A 48 ms planning step alone caps SPS at ~21. | Ch. 3, 6, 7, App. A, slides |
| M19 | CARLA "ray-traced lighting", "Boost.Asio protocol buffers", "TM density 120 veh + 60 ped /km²", "-RenderOffScreen saves 3.2 GB, >45 FPS" | CARLA 0.9.15 (UE 4.26) is rasterised; its RPC layer is rpclib/MessagePack (check CARLA docs); RL env default is `num_vehicles 3, num_walkers 10`; no log has 3.2 GB or 45 FPS. | Ch. 4, 7 |
| M20 | LiDAR 32-ch, 100 m, [-25°, +15°], at x 1.3 | TF++ config: LiDAR at (0, 0, 2.5), 600k pts/s, 10 Hz, otherwise CARLA defaults. | Ch. 4 |
| M21 | 4-Gate protocol (Table 4.2) | The protocol is real (`.agents/rules/progressive_4gate_scaling.md`) but the table mixes Atari step ranges with CARLA dollar budgets, and "RC ≥ 75%", seeds {42,137,2024}, entropy criterion for CARLA are not in the rule. | Ch. 4 |

### 3.1 Controlled facts that *are* right (keep)
`f_x = 358.5` px; "492 parameter tensors" (492/492 loaded); K = 5 waypoints at 0.5 s; exact-zero speed bin + two-hot loss; RMSNorm eps 1e-6; SwiGLU; EMA + SimSiam
projection; 601-bin support (`ez_model.py:24-25`); the Atari preset table (`atari_config.py`); `Δt = 0.05 s` (`--sim-dt` default); 5.45× → 1.52×; 13.6% (as a loss);
"73.5% of the loss"; S-024 (pure self-imitation), S-025 (clamp `min=-10`), "96% NOOP" (`NOOP:590 / FIRE:25`), S-027 (PPO clip).

---

## 4. Figures (`Thesis_typist/figures/generate_figures.py`)

| Figure (used in) | What the script does |
|---|---|
| `carla_scaling_curves.png` (Ch. 5, slides) | `0.55*exp(-epochs/12)+0.18+noise` with `RandomState(42..47)` — "**Simulated learning curves based on actual telemetry**" (line 81). Asymptotes (0.38/0.32, 0.375/0.322) are typed in; the curvature panel is two formulas. Caption presents them as measured ADE curves. |
| `bench2drive_comparison.png` (Ch. 5, slides) | Hard-codes Table 5.1 (66.9, 59.14, 56.86, 65.40, 80.33; 88.4 ... 96.5). Inherits §2.1. |
| `literature_timeline.png` (Ch. 1, slides) | Hard-coded arrays "116 papers" (26 + 57 + 33), not computed from the bib. Year counts match an older bib snapshot (e.g. <2012: 4 vs 10 now; 2019: 10 vs 13; 2022: 6 vs 8). Axis ends at 2024 while the caption says 2011-2026 and the SLR claims coverage to March 2026. |
| `atari_breakout_curves.png` (slides only) | Hard-coded; "Qwen + MCTS" ends at **58.0** at 100k while every table says 14.8; "Qwen-Tiny actor-critic" ends at 14.5; baselines typed in. |

---

## 5. Atari chapter (Ch. 6) and Ch. 2 Atari table

| Item | Finding |
|---|---|
| Five-game table (Pong 12.4, MsPacman 1180, Qbert 1420, SpaceInvaders 510, Breakout 14.8) | Only Breakout was trained at scale. Archived `E:\MThesis_EXP\imports\town01_ablation_and_100k_benchmarks_20260921\...\100k_benchmark\_logs\S048b_100k_pong_dense8epoch_s0.log`: **Pong -21.00 at every eval to 99k steps** ("Peak Eval: -21.00"). The seven archived Breakout 100k runs (S047/S048) end at 11.0 (×3), 3.0 (×2), 14.0 and 2.0 (final eval; one run peaked at 20.0). No MsPacman/Qbert/SpaceInvaders run exists. **SpaceInvaders is not one of the 26 Atari-100k games** (SPR Table, Appendix). |
| Mean HNS 44.1% / median 38.2% (Table 2.4, Ch. 6, Ch. 7) | Recomputed from the thesis's own per-game scores: per-game HNS = 93.8 / 45.5 / 13.1 / 9.5 / 23.8 → **mean 37.1%, median 23.8%** (the text lists 92.5 for Pong; mean 36.9). Progression table means recompute to 1.6 / 9.0 / 23.6 / 31.9 / 37.1 (claimed 4.2 / 12.8 / 27.4 / 37.2 / 44.1). DER column mean recomputes to 15.4% (claimed 19.4%). Ablation-table HNS wrong in 6 of 7 rows (1.2 pts is **-1.7%**, claimed 3.7%). |
| DER column values | SPR paper Table: DER **Pong -19.3, Breakout 1.9, MsPacman 1204.1, Qbert 1152.9** (thesis: -12.2, 8.4, 752, 840). |
| "OTR (van Hasselt)" row | Duplicates DER's 28.5 / 16.1. OTRainbow (Kielak 2019) is 26.4 / 20.4 and is not van Hasselt's. |
| BBF 143.0 / 86.0; EZ-V2 202.1 / 118.2 | EZ-V2 paper Table 1: BBF mean **224.7** / median **91.7**; EZ-V2 mean **242.8** / median **128.6**. EfficientZero: 194.3% / 109.0% (the 190.4% appears once in prose; abstract and table say 194.3). |
| CURL "Conv-SAC + InfoNCE" | CURL's Atari agent is a data-efficient Rainbow; SAC is its DMControl agent. |
| DER recipe (`02:621-626`) | DER paper's hyper-parameter table: **n-step 3 → 20**, replay period every 4 steps → **every step**, memory 1M → **unbounded**, min replay 1600. The thesis says "capacity 10^5, 2 updates/step, n = 3". The 16.1% median is DER's, not canonical Rainbow's. |
| Unclamped vs clamped table (614.2 → 2.14; 48.7 → 1.08; 0.08 → 1.42 nats; 1.2 → 14.8 pts) | Real S-025: PLoss swings -674 / +685 (not 614.2); S-026: the clamp broke the first collapse (**score 11 at 5k**) but the policy **re-collapsed to RIGHT by 10k (0.00)**; S-027: PPO clip stabilised the loss but still collapsed; later the root cause was the **Impala encoder's missing Kaiming init** (S-043/S-046), not log-prob explosion. "Clamped (stable) → 14.8 at 100k" has no run. The same intervention ("no clamping") appears with different numbers in two tables (1.2 / 48.7 / 0.08 vs 1.8 / 38.2 / 0.12), and the "Post-LN + GELU" row equals the "unclamped" column. |
| MCTS ablation M = 10/25/50/100 (12-94 ms, 8.4-15.2) | No such sweep. Collection sims were 16/32 (S-053), Gate-2 settled at 32 (S-054); EZ-V2 itself uses 16 sims on Atari (K = 8). The MCTS prototype scored 0-1 (below random 1.7) until the S-049 replay-interleave bug was fixed. |
| SVD spectrum, MCTS depth/entropy, attention-entropy tables | No script output matches (`probe_layerwise_variance.py` / `diagnose_policy_collapse.py` measure other things). Also uses d = 512 while the Atari model has d = 256. |
| "Standard sticky actions p = 0.25" / "10 evaluation seeds" | Atari-100k's standard protocol has **no** sticky actions; in this project they were introduced only for B2 (deterministic evals gave ~4 distinct games per checkpoint, memory `b2-needs-sticky-actions`). The archived 100k evals are deterministic ("11.00 +/- 0.00"). |
| Children learn Breakout "in 15-30 min (<10,000 frames)" | 15 min at 60 fps is 54,000 frames; uncited. |
| S-034 cited as "crawling equilibria" (Ch. 7) | S-034 is "on-policy GTrXL+EZ2 rewrite, flat 0.00". |
| Best real Atari results (not in the thesis) | EZ-V2 port, Breakout: GTrXL **371.6 @70k**, ResNet 377.8 @80k (EZ-V2 321.2 @100k, paper 400.1), one seed; at 10k uniform replay 15.0 vs EZ-V2 9.8; GTrXL+uniform 38.6 @10k (> human 30.5). Caveat: the GTrXL mixer was dead, so GTrXL-vs-ResNet compared ResNet with ResNet (S-116). |

---

## 6. Literature facts (Ch. 1-2) — accuracy

Format: line (in `01_introduction.typ` / `02_related_work.typ`) → issue.

**Chapter 1**
* `:75-78` WHO 1.19 M deaths, 20-50 M injuries, leading cause 5-29 y: correct (WHO 2023) but **uncited**; ">90% human error" is NHTSA's 94% "critical reason" figure, uncited and often misread.
* `:86-87` "Road transportation contributes **over 20%** of global GHG": overstated. Our World in Data/IEA: road ≈ 15% of CO₂ (≈12% of all GHG); all transport ≈ one-fifth of CO₂ from fuel combustion. `:89-91` "10-25% fleet saving" uncited (eco-driving literature mostly 5-20%); Ch. 7 says 10-15%.
* `:57-60` LiDAR "$10k-50k, mechanical **and solid-state**": outdated; solid-state units are now hundreds of dollars.
* `:46` SAE L5 "eliminating driver controls" is not in J3016; `:48-50` "CARLA challenge and Bench2Drive target Level 4" is not stated by either.
* `:128` "Ross & Bagnell proved" cites Ross, Gordon & Bagnell 2011 (the T²ε result is Thm 2.1 there / Ross & Bagnell 2010).
* `:143` Qwen/Qwen2 text-LLM reports cited for "multimodal spatial representation"; `:147` `levine2016visuomotor` and `koh2021wilds` cited as evidence that scaling transformers generalises robot manipulation (neither is about that).
* `:149-150` **TCP is described as a transformer fusing LiDAR and multi-camera**: TCP is a single-front-camera CNN + GRU, no LiDAR (it is the closest published single-camera baseline).
* `:158` "over 110 peer-reviewed works" (bib 130, cited 88, SLR 126, figure 116; several are preprints, reports, theses).
* Two different sets of RQ1-RQ4 (Ch. 1 vs Ch. 2).

**Chapter 2**
* PRISMA flow (`:57-62`): 642 → 488 (-154) → -294 = 194 → -78 = **116**, text says 126. No search log, query output or screening sheet exists in the repo; the only corpus file is `Thesis_typist/references.csv` (165 titles with Semantic Scholar citation counts). IC1 ("peer-reviewed in premier venues") and the 2011-2026 window are violated by the bibliography (10 entries before 2012 incl. ALVINN 1989; arXiv-only Qwen, OpenVLA, RT-1/2, TF++, EZ-V2; two bachelor/diploma theses; **no entry from 2025-2026**). The quality criteria are "adapted from Kitchenham & Charters" but the cited key is the 2004 single-author report.
* `:136` ALVINN "single-layer perceptron" contradicts `:158-159` ("three-layer"); ALVINN had one hidden layer.
* `:137` CILRS "command-conditioned **waypoint** prediction": CIL/CILRS output steer/throttle/brake (also `:196-197`).
* `:212-243` the "formal inductive proof" bounds the learner's per-step error under *its own* state distribution by ε (which is defined under the expert's) — circular; and `:241-243` an upper bound of 10,000 on a 1,000-step episode is vacuous and cannot "guarantee" failures.
* `:290` AggreVaTe = "Aggregate Values to **Imitate**", and it is not a policy-gradient method (AggreVaTeD is).
* `:354` LSS authors are Philion **& Fidler** (arXiv 2008.05711), not "Baker".
* `:396-400` TransFuser/TF++ rasterise LiDAR into a BEV height histogram; they do not use PointPillars. TF++ in Table 2.5: encoders are **RegNetY-032** (not ResNet-34), 8 waypoints over 2.0 s (not 4) (`Carla-utils/carla_garage/team_code/config.py:378,466`); and Ch. 3 itself takes the backbone from the TF++ checkpoint.
* `:330-334` "TF++ and InterFuser achieve DS > 80": TF++ Bench2Drive 84.21 yes; InterFuser LB1.0 76.18 no.
* Table 2.3 (latency/VRAM of LSS, BEVDepth, BEVFormer, SimpleBEV, TF++, UniAD "on an RTX 3090") has no measurement script; BEVDepth, SimpleBEV, ChauffeurNet, Wayformer, GAIA-1, Deformable DETR, PointPillars, Ziebart (MaxEnt IRL), SafeDAgger, AggreVaTe are named without a citation (`ho2016gail` exists in the bib but the text names "Ho & Ermon" without citing it).
* `:440` the RSSM was introduced in PlaNet, not Dreamer; `:464` DreamerV2 does not claim to fix "posterior collapse"; `:469` "completely averting latent collapse ... 3D spatial stability" is unsupported; `:485` "10× higher simulation throughput" uncited.
* `:503-505` not all of RT-1/PaLM-E/SayCan "discretise into 256 tokens ... multi-billion-parameter transformers" (RT-1 is 35M, PaLM-E and SayCan do not emit action tokens).
* `:540` RMSNorm paper reports **7-64% running-time** reduction, not "7% memory bandwidth".
* `:551-553` "learnable residual skip multipliers" are not a Qwen/LLaMA component (they are ReZero/LayerScale-style); Qwen2's real features (GQA, QKV bias) are omitted.
* Table 2.2: GTrXL "limitation under 100k" and Decision Transformer "poor Atari-100k efficiency": neither paper evaluated Atari-100k; these are the thesis's own findings written as literature limitations.
* `:651-672` the four "fundamental failure modes of transformers in RL" are uncited; (4) misdescribes PPO ("clipping individual action logits").
* `:683-686` "GPUs spend over 85% waiting on RPC" and `:719-721` ">90% GPU duty cycle": the log says `world.tick()` is 81-85% of `step()` wall time at ~22% GPU utilisation (challenges_04 §4.3-4.4), a different statement; no log shows >90%.
* `:30-35` the "five databases" (IEEE Xplore, ACM DL, Scopus, ...) and Boolean queries have no record.

---

## 7. Reference list

### 7.1 Counts
130 entries; **88 cited** in the compiled thesis; **42 never cited anywhere** (printed anyway because of `full: true`; the guidelines stress citing every source used, and 42 listed-but-uncited works read as padding);
0 cited keys missing. The bib header says "Total verified entries: 85+" — that statement is false.

### 7.2 Cited entries with wrong bibliographic facts (14)

| Key | In the bib | Verified |
|---|---|---|
| `jaeger2023transfuserplus` | "Hidden Gems in End-to-End Driving", arXiv:2307.08704 | **arXiv:2307.08704 is an astrophysics paper.** TF++ = "Hidden Biases of End-to-End Driving Models", Jaeger, Chitta, Geiger, ICCV 2023, arXiv:2306.07957. Cited ≈10×, incl. as the baseline and the backbone source. |
| `ye2024efficientzerov2` | "EfficientZero V2: Master of Discrete and Continuous Environments", Ye, Zhang, Abbeel, Gao, arXiv:2403.00572 | arXiv:2403.00572 is an astrophysics paper. Real: "EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data", Wang, Liu, Ye, You, Gao, ICML 2024, arXiv:2403.00564. |
| `gao2024bench2drive` | Gao, Qiu, Song et al., CoRL 2024 | Jia, Yang, Li, Zhang, Yan; "...Closed-Loop **End-To-End** Autonomous Driving"; **NeurIPS 2024 Datasets & Benchmarks**; arXiv:2406.03877 (your own `eval_tiers_design.md:362` already links the NeurIPS PDF). |
| `chen2022tcp` | Penghao Chen, Juntian Zheng, Yilun Chen, Jiaming Liu, Yitao Liu, Hao Zhao | Penghao **Wu**, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, Yu Qiao; full title ends "...: A Simple yet Strong Baseline"; NeurIPS 2022; arXiv:2206.08129 (pages not verified). |
| `dauner2024pdmlite` | "Parting with Misconceptions about Learning-based **Driving**", CoRL **2024** | "...Learning-based **Vehicle Motion Planning**", CoRL **2023**, arXiv:2306.07962 — the nuPlan PDM-Closed paper. PDM-Lite (CARLA) is cited by its dataset card as J. Beißwenger, "PDM-Lite: A Rule-Based Planner for CARLA Leaderboard 2.0" (Tübingen 2024) and DriveLM (ECCV 2024); the Hidden-Biases-of-Datasets report (Zimmerlin et al., arXiv:2412.09602) also describes it. |
| `shao2023interfuser` | 6 authors incl. Steven L. Waslander; CoRL **2023** | 5 authors (Shao, Wang, Chen, Li, Liu); **CoRL 2022** (PMLR 205, published 2023); arXiv:2207.14024. Waslander is on ReasonNet. |
| `chen2021wor` | CoRL 2021 | **ICCV 2021 (Oral)**, arXiv:2105.00636. This is the baseline paper. |
| `hafner2022dreamerv2` | "Mastering Diverse Domains through World Models", ICLR 2022 | That title is DreamerV3's. DreamerV2 = "Mastering Atari with Discrete World Models", ICLR **2021**, arXiv:2010.02193. The list prints two entries with the same title ([31], [32]). |
| `hu2022mile` | "Model-based Imitation Learning for End-to-End AD", Shengchao Hu, Li Chen, ... Dacheng Tao | Those authors wrote ST-P3 (ECCV 2022). MILE = "Model-Based Imitation Learning for **Urban Driving**", Anthony Hu, Corrado, Griffiths, Murez, Gurau, Yeo, Kendall, Cipolla, Shotton, NeurIPS 2022, arXiv:2210.07729. |
| `wang2024drivewm` | "Drive-WM: World Models for Autonomous Driving", Xiaofeng Wang, Zheng Zhu, ... Jiwen Lu | Those authors wrote DriveDreamer. Drive-WM = "Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving", Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, Zhaoxiang Zhang, CVPR 2024, arXiv:2311.17918. |
| `rosin2011puct` | "Multi-armed bandits with episode **bonuses**", JAIR 41:329-368 | "Multi-armed bandits with episode **context**", *Annals of Mathematics and AI* 61(3):203-230, 2011, doi:10.1007/s10472-011-9258-6 (Crossref). |
| `bojarski2016end` | co-authors include "Delia Isele" | arXiv:1604.07316 has 13 authors, none named Isele (Del Testa, Dworakowski, Firner, Flepp, Goyal, Jackel, Monfort, Muller, Zhang, Zhang, Zhao, Zieba). Print shows "D. D. Testa" (should be "D. Del Testa"). |
| `vlachogiannopoulos2021ntua` | "Deep RL for Unknown Transition Models with Applications to Atari Video Games", Master's thesis | NTUA DSpace (handle 123456789/54605): "Deep Reinforcement Learning Using Exploration Methods Based on **Curiosity Reward Signals**", M. Vlachogiannopoulos, 26 Oct 2021, ECE diploma thesis. The title in the bib was not found. The Ch. 6 sentence ("highlighted the extreme sensitivity of transition modeling to perceptual feature extraction") does not describe it. |
| `vondikakis2021ntua` | "Deep RL Methods Applied to Video Games", 2021 | NTUA DSpace (123456789/56169): "Methods of Deep Reinforcement Learning in the Video Game **Super Mario Bros**", I. Vondikakis, 6 Jul **2022**, undergraduate thesis. Not Atari; same Ch. 6 sentence unsupported. |

Minor / format defects on cited entries: `hu2023uniad` lists a non-author ("Shenxian Deng"; UniAD's 7th author is Siqi Chai, arXiv:2212.10156); `jiang2023vad` omits Chang Huang from a list that has no `and others`; `brohan2022rt1` lists "Mahdi M. Nejad" (not among RT-1's 51 authors); `mnih2016a3c` has broken accent macros (`Adri{\}`, `{\e}`); `li2022bevformer` prints "Birdś" (`\'s`); the two NTUA theses print without institution; `chen2020lbc` is CoRL 2019 (PMLR vol. 2020); `chitta2022transfuser` TPAMI vol. 45 is 2023; `kitchenham2004guidelines` is cited as "Kitchenham & Charters" (EBSE-2007-01 is the Charters report); entries with `and others` print literally "and others" (e.g. [5], [24], [39]) instead of "et al.".

### 7.3 Uncited entries (42) — and 9 of them are wrong
Wrong facts: `prakash2021transfuser` (CVPR 2021, not CoRL), `zhang2021challenges` (authors/venue of Roach; real paper is Chen et al., TPAMI 2024), `toromanoff2020end` (CVPR 2020, not T-ITS), `filos2020can` (ICML 2020, title "...Adapt to Distribution Shifts?"), `badia2020agent57` (ICML 2020, not Nature), `spencer2021feedback` ("...The Three Regimes of Covariate Shift", arXiv:2102.02872, not ICML 2021).
Format: `pomerleau1991efficient` is `@inproceedings` with a `journal` field, so the journal name is lost in print ([115] shows no venue); `kingma2014adam` is an `@article` whose journal is ICLR; `tampuu2020survey` has 2022 volume/pages but year 2020.
Full list: ba2016layernorm, badia2020agent57, burda2019rnd, caesar2020nuscenes, castro2018dopamine, chitta2021neat, engstrom2020implementation, filos2020can, fujimoto2018td3, geiger2012kitti, haarnoja2018sac, hausknecht2015drqn, hawke2020urban, he2016resnet, hester2018dqfd, ho2016gail, kingma2014adam, kumar2020cql, levine2020offline, lillicrap2016ddpg, liu2021swin, loshchilov2019adamw, machado2018revisiting, micikevicius2018mixed, paszke2019pytorch, pathak2017curiosity, pomerleau1991efficient, prakash2021transfuser, schulman2015trpo, shao2023reasonnet, silver2014dpg, silver2016alphago, silver2017alphazero, spencer2021feedback, sun2020waymo, tampuu2020survey, toromanoff2020end, touvron2023llama, williams1992reinforce, yarats2022drqv2, zaharia2018mlflow, zhang2021challenges.
(`silver2017alphazero` is AlphaGo Zero; the AlphaZero paper is Science 2018.)

### 7.4 Checked and fine
The bulk of the RL/vision/LLM classics match arXiv/Crossref/OpenAlex on title, first author and year: DQN (both), Double-Q, Dueling, PER, Rainbow, ALE (both), Dopamine, TRPO, PPO, SAC, DDPG, TD3, IMPALA, R2D2, World Models, PlaNet, Dreamer, DreamerV3, AlphaGo, AlphaGo Zero, MuZero, EfficientZero, SPR, CURL, DrQ, DrQ-v2, ViT, Swin, GTrXL, LLaMA, Qwen, Qwen2, GLU variants, RMSNorm, RoFormer, ResNet, RegNet, BatchNorm, LayerNorm, AdamW, Adam, DAgger, DART, CQL, BCQ, BEAR, GAIL, DQfD, MLflow, PyTorch, mixed precision, RT-1/RT-2/PaLM-E/OpenVLA/SayCan, WILDS, CARLA, CIL, CILRS, LBC, TransFuser (TPAMI), NEAT, UniAD, VAD, ReasonNet, nuScenes, Waymo, KITTI, Decision/Trajectory Transformer, DER ("When to use parametric models"), BBF, DeepDriving, causal confusion, BEVFormer. Page numbers for these were spot-checked from memory only.

### 7.5 Corrected entries (fields verified against arXiv/Crossref; pages deliberately omitted where unverified)

```bibtex
@inproceedings{jaeger2023hiddenbiases,
  author = {Bernhard Jaeger and Kashyap Chitta and Andreas Geiger},
  title = {Hidden Biases of End-to-End Driving Models},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)}, year = {2023}, note = {arXiv:2306.07957} }
@inproceedings{jia2024bench2drive,
  author = {Xiaosong Jia and Zhenjie Yang and Qifeng Li and Zhiyuan Zhang and Junchi Yan},
  title = {{Bench2Drive}: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track}, year = {2024}, note = {arXiv:2406.03877} }
@inproceedings{wu2022tcp,
  author = {Penghao Wu and Xiaosong Jia and Li Chen and Junchi Yan and Hongyang Li and Yu Qiao},
  title = {Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2022}, note = {arXiv:2206.08129} }
@inproceedings{shao2022interfuser,
  author = {Hao Shao and Letian Wang and Ruobing Chen and Hongsheng Li and Yu Liu},
  title = {Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer},
  booktitle = {Conference on Robot Learning (CoRL)}, year = {2022}, note = {PMLR 205:726--737; arXiv:2207.14024} }
@inproceedings{chen2021wor,   % venue fix
  author = {Dian Chen and Vladlen Koltun and Philipp Kr{\"a}henb{\"u}hl},
  title = {Learning to Drive from a World on Rails},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)}, year = {2021}, note = {Oral; arXiv:2105.00636} }
@inproceedings{prakash2021transfuser,   % venue fix
  author = {Aditya Prakash and Kashyap Chitta and Andreas Geiger},
  title = {Multi-Modal Fusion Transformer for End-to-End Autonomous Driving},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year = {2021}, note = {arXiv:2104.09224} }
@inproceedings{hu2022mile,
  author = {Anthony Hu and Gianluca Corrado and Nicolas Griffiths and Zak Murez and Corina Gurau and Hudson Yeo and Alex Kendall and Roberto Cipolla and Jamie Shotton},
  title = {Model-Based Imitation Learning for Urban Driving},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2022}, note = {arXiv:2210.07729} }
@inproceedings{wang2024drivewm,
  author = {Yuqi Wang and Jiawei He and Lue Fan and Hongxin Li and Yuntao Chen and Zhaoxiang Zhang},
  title = {Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year = {2024}, note = {arXiv:2311.17918} }
@inproceedings{wang2024ezv2,
  author = {Shengjie Wang and Shaohuai Liu and Weirui Ye and Jiacheng You and Yang Gao},
  title = {{EfficientZero V2}: Mastering Discrete and Continuous Control with Limited Data},
  booktitle = {International Conference on Machine Learning (ICML)}, year = {2024}, note = {arXiv:2403.00564} }
@inproceedings{hafner2021dreamerv2,
  author = {Danijar Hafner and Timothy Lillicrap and Mohammad Norouzi and Jimmy Ba},
  title = {Mastering Atari with Discrete World Models},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2021}, note = {arXiv:2010.02193} }
@inproceedings{dauner2023pdmclosed,
  author = {Daniel Dauner and Marcel Hallgarten and Andreas Geiger and Kashyap Chitta},
  title = {Parting with Misconceptions about Learning-based Vehicle Motion Planning},
  booktitle = {Conference on Robot Learning (CoRL)}, year = {2023}, note = {arXiv:2306.07962} }
@misc{beisswenger2024pdmlite,   % as recommended by the dataset card autonomousvision/PDM_Lite_Carla_LB2
  author = {Jens Bei{\ss}wenger}, title = {{PDM-Lite}: A Rule-Based Planner for {CARLA} Leaderboard 2.0},
  howpublished = {University of T{\"u}bingen technical report}, year = {2024} }
@article{rosin2011episode,
  author = {Christopher D. Rosin}, title = {Multi-armed bandits with episode context},
  journal = {Annals of Mathematics and Artificial Intelligence}, volume = {61}, number = {3}, pages = {203--230}, year = {2011} }
@article{chen2024e2esurvey,
  author = {Li Chen and Penghao Wu and Kashyap Chitta and Bernhard Jaeger and Andreas Geiger and Hongyang Li},
  title = {End-to-end Autonomous Driving: Challenges and Frontiers},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, year = {2024}, note = {arXiv:2306.16927} }
@inproceedings{philion2020lss,
  author = {Jonah Philion and Sanja Fidler},
  title = {Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3{D}},
  booktitle = {European Conference on Computer Vision (ECCV)}, year = {2020}, note = {arXiv:2008.05711} }
@inproceedings{toromanoff2020implicit,  % uncited today
  author = {Marin Toromanoff and Emilie Wirbel and Fabien Moutarde},
  title = {End-to-End Model-Free Reinforcement Learning for Urban Driving using Implicit Affordances},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, year = {2020}, note = {arXiv:1911.10868} }
@inproceedings{filos2020distshift,      % uncited today
  author = {Angelos Filos and Panagiotis Tigas and Rowan McAllister and Nicholas Rhinehart and Sergey Levine and Yarin Gal},
  title = {Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?},
  booktitle = {International Conference on Machine Learning (ICML)}, year = {2020}, note = {arXiv:2006.14911} }
@inproceedings{badia2020agent57,        % uncited today
  author = {Adri{\`a} Puigdom{\`e}nech Badia and Bilal Piot and Steven Kapturowski and Pablo Sprechmann and Alex Vitvitskyi and Zhaohan Daniel Guo and Charles Blundell},
  title = {Agent57: Outperforming the {Atari} Human Benchmark},
  booktitle = {International Conference on Machine Learning (ICML)}, year = {2020}, note = {arXiv:2003.13350} }
@misc{spencer2021threeregimes,          % uncited today
  author = {Jonathan Spencer and Sanjiban Choudhury and Arun Venkatraman and Brian Ziebart and J. Andrew Bagnell},
  title = {Feedback in Imitation Learning: The Three Regimes of Covariate Shift}, year = {2021}, note = {arXiv:2102.02872} }
@mastersthesis{vlachogiannopoulos2021curiosity,   % NTUA DSpace 123456789/54605 (diploma thesis)
  author = {Marios Vlachogiannopoulos},
  title = {Deep Reinforcement Learning Using Exploration Methods Based on Curiosity Reward Signals},
  school = {National Technical University of Athens, School of Electrical and Computer Engineering}, year = {2021} }
@mastersthesis{vondikakis2022mario,               % NTUA DSpace 123456789/56169
  author = {Ioannis Vondikakis},
  title = {Methods of Deep Reinforcement Learning in the Video Game Super Mario Bros},
  school = {National Technical University of Athens, School of Electrical and Computer Engineering}, year = {2022} }
```
For `bojarski2016end` use the real author list ("... Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, ..." with `and others`).
If the two NTUA theses are kept, they need a sentence that matches what they cover (curiosity-driven exploration; Super Mario Bros) or should be dropped.

### 7.6 Cited works missing from the bib
SAE J3016; WHO *Global status report on road safety 2023*; NHTSA critical-reasons study; an eco-driving source; a LiDAR cost source; Ziebart et al. 2008 (MaxEnt IRL); Zhang & Cho (SafeDAgger); Ross & Bagnell 2014 (AggreVaTe); Philion & Fidler (LSS); Zhu et al. (Deformable DETR); Lang et al. (PointPillars); BEVDepth; Simple-BEV; ChauffeurNet; Wayformer; GAIA-1; Kitchenham & Charters 2007; a citation for CARLA Garage / PDM-Lite (see above). Also, no reference postdates 2024, so the thesis never positions itself against SimLingo (single front camera, DS 85.1), CarLLaVA, LEAD/TFv6 (95.0), PlanT 2.0, or Fail2Drive, all of which your own `what_has_worked_2026-10-04.md` cites.

---

## 8. What checks out

* Intro: WHO statistics (needs a citation), SAE levels 0-5, POMDP formulation, the O(εT²) behaviour-cloning bound (as a statement; see §6 for the proof), forward/reverse KL discussion, MaxEnt-IRL and GAIL formulas, PUCT formula (MuZero form), DT/TT/GTrXL/RT-2/OpenVLA descriptions, ALVINN input/output sizes, SPR/SimPLe/CURL/DrQ/DER aggregate HNS, Atari random/human scores for the four listed games, the Atari-100k definition (100k agent steps = 400k frames ≈ 2 h).
* Method facts listed in §3.1.
* The Bench2Drive metric *form* (DS = mean of RC × IS), the multiplicative penalty structure, and coefficients 0.6 / 0.65 / 0.7 / 0.8 (vehicle, static, red light, stop). Wrong: pedestrian collision is **0.50** (thesis: 0.0 "DS = 0"), outside-route-lanes is a **continuous** penalty (thesis: 0.9 per occurrence), and collisions do **not** terminate a route (only `ROUTE_DEVIATION` and `VEHICLE_BLOCKED` do; `statistics_manager.py:21-52, 385-404`). Bench2Drive routes are ~150 m each (thesis: D_i ∈ [200 m, 1.5 km]).
* The 4-Gate protocol exists as a project rule; scripts named in Ch. 4 exist (`scripts/setup/remote_config.py`, `scripts/sync/sync_experiments.py`); `E:\MThesis_EXP` is real.
* The challenges appendix is mostly a faithful condensation of the logs: groups 1 (uid-0 refusal, `unshare` failure), 2 (`pkill` self-kill), 3, 4 (1.4-1.7-tick staleness), 9 (predict-zero trap), 11 (camera mismatch), 13 (retractions, 5.45× → 1.52×). Embellishments to remove: Group 4 resolution (shared-memory rings; the log's fix is `--sim-dt` + N shared servers), Group 5 (68% / 7.2→3.8 GB / `taskset` cores, 42% in Ch. 3 contradicts 68%), Group 6 (Pydantic schema), Group 8 snippet (log: override `train()`; the snippet sets `track_running_stats=False`, which makes BN use batch statistics whenever `train()` is called again), Group 9 resolution ("repaired the data-collection harness" — the data are the public PDM-Lite set whose command enum is unset by design; the fix was route conditioning), Group 10 (`np.memmap`, 72% `cv2.imdecode`, 98.4%: log says 17% → 42% → 99% GPU util, PIL decode, `.npy` cache), Group 12 ("28 tests ... 15 s"; `src/data/` does not exist; the suite now has 374 test functions in 55 files), Group 13 ("five-fold from 214 routes": 129 → 656 is 5×, 214 → 656 is 3×), the "eight-week campaign" (git history 2026-07-17 → 2026-10-05, 402 commits), and two different HF repo names (`miaris98/MThesis` in Ch. 4, `miaris98/MThesis-Checkpoints` in Group 2; the scripts default to `Miaris/mthesis-wor-checkpoints`).

---

## 9. Suggested repair order

1. **Decide the results story from records, then rewrite Ch. 5/6 around it.** Use: arm H vs E (S-090), A9 on 220 routes with paired CIs (S-108/S-113/S-115, `a9_merge.py`, `E:\MThesis_EXP\analysis_*`), the ledger's grades (`what_has_worked_2026-10-04.md`), 13.20/13.23 (as *loss*, three seeds, 10.8-13.6%), 5.45× → 1.52×, route conditioning, the multi-town data effect, and for Atari the EZ-V2 port numbers with the "dead mixer" caveat (S-116). Report geometry loss and recovery cameras as "tried, not established".
2. **Delete or clearly mark as not run**: Tables 5.3-5.5, 5.7, 5.9-5.14, case studies, Jetson/TensorRT, Atari multi-game, M-sweep, SVD/attention/search tables, 18 → 74 SPS. If a table is kept as a design hypothesis, label it as such.
3. **Regenerate every figure from real telemetry/CSV** (`data/wor_training_telemetry_qwen100.csv` and the E: archive exist); stop using `RandomState` curves; make the timeline read the bib.
4. **Align Ch. 3/4 and Appendix A with the code** (§3): RoPE → learned position/type embeddings; autoregressive → one-shot head; 8 layers / ffn 1536; 20-token sequence incl. the route token and overlay; real camera and crop; TF++ speed bins; the real PID; the real loss; real optimiser/precision; dataset provenance (public PDM-Lite, real town lists and frame counts); recovery augmentation as DataAgent views + rigid re-projection.
5. **Fix the bibliography** (§7): corrected entries above, drop or cite the 42 uncited ones (set `full: false`), add the missing citations and the 2025-26 camera-only/LEAD/SimLingo line of work, repair the two syntax/format defects.
6. **Rewrite the SLR section honestly**: either supply the actual search log and counts, or describe the corpus as a non-systematic literature survey (`references.csv`).
7. Fix the literature misstatements in §6 and the Bench2Drive/Leaderboard facts in §8.
8. Finally align the abstract, introduction contributions, Ch. 7 answers to RQ1-4, the slides (`slides.typ:135, 212-226, 259-262, 289-291, 334, 362-368, 394-395`) and the two RQ lists.

Open items I cannot settle from the repo: whether any of the §2.3 tables came from runs on a box that was destroyed before its logs were synced (2026-09-14 and 2026-09-30 destroys are recorded); whether `miaris98/MThesis*` exist on Hugging Face; page numbers of a few classic papers (DBLP was unreachable).

---

## 10. Repair status (2026-10-05, evening — "lets do both": chapters 5-6, figures and bibliography, plus the chapters that depend on them)

`Thesis_typist/` is in `.gitignore`, so the pre-repair state is only in the copy `C:\Users\miari\Desktop\Thesis_typist_pre-repair_2026-10-05\` (215 files). Nothing was committed.

| Area | What was done |
|---|---|
| Bibliography | `bib/references.bib`: 15 keys renamed and every citation rewritten, the 14 wrong cited entries and 9 wrong uncited ones corrected, author lists completed from arXiv (no more "and others" in print), 26 verified entries added (SimLingo, CarLLaVA, LEAD, PlanT 2.0, Safe2Drive, DriveLM, LSS, ReZero, LayerScale, He init, Gumbel MuZero, AggreVaTe, MaxEnt IRL, SafeDAgger, Deformable DETR, PointPillars, BEVDepth, Simple-BEV, ChauffeurNet, Wayformer, GAIA-1, SAE J3016, WHO 2023, NHTSA, Kitchenham & Charters). `bibliography(..., full: false)`: only the 86 cited works print. Unused entries stay in the file. |
| Figures | `figures/generate_figures.py` rewritten: no random numbers; every constant is transcribed from a named log entry; the timeline is computed from the bibliography. New: b2d_main_result, b2d_group_effects, intervention_effects, points_lost, abilities, heldout_loss, atari_breakout, policy_architecture, literature_timeline. The three synthetic PNGs were deleted. E: was not mounted, so values come from the logs, not from raw telemetry. |
| Ch. 5 | Rewritten around S-117/S-120/S-108/S-128 and 13.x: E 63.4 vs WoR 49.4 (+13.7 [+8.6, +18.8]), group and ability breakdown, held-out loss as *loss* (10.8-13.6%), evidence ledger with grades, obstacle trade-off, failure anatomy, threats to validity. Tables 5.3-5.5, 5.7, 5.9-5.14, case studies and Jetson/TensorRT profiling were removed (no run). |
| Ch. 6 | Rewritten: collapse root-cause chain (S-024 to S-046), prototype bugs and the EZ-V2 port (S-049 to S-104), dead GTrXL mixer (S-116/S-129); the multi-game table, M sweep, SVD/attention tables, 18 to 74 SPS were removed. |
| Ch. 3, 4, App. A (code) | Ch. 3 describes the code (no RoPE, one-pass heads, 8 layers / ffn 1536, 20 tokens, route input, camera, TF++ bins, PID on heading angle, the real loss, the DataAgent recovery view with rigid re-projection). Ch. 4: public PDM-Lite data, real subsets and counts, arm table, defect catalogue, Atari protocol, gate rule as written, measured compute. Appendix B listings are generated verbatim from the repository (commit 73b259a). |
| Ch. 1, 2, 7, abstracts, App. A (challenges), slides | RQs, contributions and conclusions rewritten to match; Ch. 2 is now a narrative review (PRISMA numbers removed; corrected proof of the O(εT²) bound; literature facts fixed); the challenges appendix is rebuilt from the logs; slides rewritten (10 slides, real figures). |
| Template | `ihu_template.typ`: a `show ref` rule for figures and tables (cross-chapter references printed the referencing chapter's number, e.g. "Table 5.2" for a table in Chapter 4). |
| Build | 83 pages (was 114), compiles with no warnings; `MIT-Thesis.pdf` and `presentation/presentation.pdf` rebuilt. |

Decisions that stay with the author: (1) the SLR section is now a narrative review because no search log exists; if the PRISMA search was really run, restore it from the export; (2) the Greek abstract is my translation of the English one and needs a native read; (3) title page metadata, declaration, acknowledgments and the date (September 30, 2026) were not touched; (4) the text does not mention AI assistance, while the logs do — check the institution's rule; (5) claims that would need new runs are listed as future work in Ch. 7 (blank-image ablation, strict held-out-town protocol, latency, multi-game Atari).
