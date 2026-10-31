---
trigger: always_on
description: Thesis targets (beat TF++ and EZ-V2), shared Qwen30m architecture across CARLA and Atari, CARLA-pretrained frozen vision backbone, RGB-only sensor constraint, and invalidation guards.
---

# Thesis Scope, Shared Qwen30m Architecture & Baseline Rules

## 1. Fixed Thesis Headline Targets (2026-10-05 / 2026-10-07)
The thesis goals are fixed by the user and must **never** be reframed into narrower/gap-analysis claims:
1. **CARLA**: Beat **TransFuser++ (TF++)** on the official 220-route Bench2Drive benchmark (TF++ published 84.2 DS; 80.33 DS on our 19-route `b2d20` subset) using a single front RGB camera and frozen CARLA-pretrained backbone, evaluated with `>= 3` runs and paired bootstrap 95% CI. (World on Rails is already beaten on 220 routes: Arm E15 63.4 vs original WoR 49.4, paired `+13.7 [+8.6, +18.8]`, S-117).
2. **Atari-100k (Breakout)**: Beat **EfficientZero-v2 (EZ-V2)** (`400.1` paper score; `363.1` @50k / `337.9` @120k final in our official reproduction) with `>= 5` seeds at 100k env steps (`120,000` updates) under the paper protocol (16 sims, 60 episodes, no sticky actions, no FIRE reset), alongside a sticky-action (`p=0.25`) robustness column.
- State odds honestly when planning, then plan backwards from winning both targets. Never claim a win from a single seed or partial run.

---

## 2. Exact Same `qwen30m` Network on CARLA and Atari (2026-10-25 / 2026-10-26)
- **User Directive**: Qwen must beat EZ-V2 while using the **EXACT SAME network** on both CARLA and Atari (`policy_arch qwen30m` / `MIXER_SIZES["30m"]`: `8` `QwenTransformerBlock`s, `d_model=512`, `8` heads, `d_ff=1536`, `~25.2M` block params). Do not exceed `30m` (100m is flat on CARLA and ~100h/seed on Atari).
- **Atari Architectural Layout (`--mixer-where pred`, TODO B69, 2026-10-26)**:
  - EZ-V2's convolutional representation + dynamics network plays the CNN encoder's role; the `qwen30m` trunk is the **prediction function** in CARLA's `QwenWaypointTransformer` layout (one token per `6x6` state cell with learned position, a learned policy token in front excluded from weight decay, `QwenTransformerBlock`s + final `RMSNorm`, and `Linear-GELU-Linear` value/policy heads reading from the policy token).
  - Note: `--mixer-where repr` (D1, Qwen in representation only) also remains in the codebase as a user-approved option that requires no CARLA rerun unless `QwenTransformerBlock` itself changes.
- **Shared Block Coupling Rule**:
  - If `QwenTransformerBlock` is ever modified (e.g., adding Qwen3-style per-head **QK-norm** via `--qk_norm 1` / `--mixer-qk-norm`, TODO B67), **CARLA's best arm (E15) MUST be rerun with that exact modification** before any Atari score using it can count as a thesis claim.
  - Training-recipe changes (such as optimizer/LR groups) do not alter the block definition and do not invalidate CARLA E15.

---

## 3. Vision Pretrained Scope, Frozen Backbone & RGB-Only Sensor Rules
- **Never Train or Fine-Tune Vision Models (`freeze_backbone=True`)**:
  - Training vision backbones from scratch or fine-tuning them is out of scope. Vision backbones must remain **frozen** (`requires_grad=False`) AND explicitly placed in **`.eval()` mode** during training so `BatchNorm` running statistics (`running_mean`/`running_var`) do not drift.
- **CARLA-Pretrained Backbones Only (Never ImageNet)**:
  - Vision backbones for CARLA must be pretrained on CARLA driving data, never generic ImageNet (`pretrained=True` in `timm`/`torchvision` is forbidden).
  - Sanctioned sources: (1) `regnety_032` (`18.0M` params) extracted from the `carla_garage` TransFuser++ release (`pretrained_models.zip`), which is also weather-robust; (2) the user-approved CARLA-fine-tuned `InternVL2-1B` vision tower from SimLingo (`RenzKa/simlingo`, TODO A65–A67, loaded with `weights_only`). Any non-CARLA VLM encoder requires explicit user approval.
- **RGB-Only Policy Input (No Privileged Sensors)**:
  - Policy inputs are restricted to **front RGB camera** (plus ego speed, route/target point, driving command). Never add LiDAR or ground-truth BEV as network inputs.
  - Extra modalities (`depth`, `semantics`, `bev_semantics`, `boxes`) are permitted **only** as auxiliary training targets predicted from RGB, or derived at test time from RGB predictions (e.g., pseudo-LiDAR lifted from predicted depth).

---

## 4. World on Rails Baseline & CARLA Dataset / Camera Rules
- **World on Rails (`wor`) Baseline Integrity**:
  - `--reward-fn=wor` and `--policy-arch=wor` represent the comparison baseline (`Carla-utils/PCLA/pcla_agents/wor/`). Do not hand-tune `wor` to train better; any fix must be justified strictly by fidelity to upstream `wor/rails/bellman.py`.
  - Always distinguish: (a) the released PCLA `wor_lb` agent (4 RGB cameras, 66.9 DS on `b2d20`, 49.4 DS on Bench2Drive-220) vs (b) Arm H (WoR's `cnn` head trained in our 1-camera pipeline).
- **Camera Geometry & Dataset Compatibility**:
  - Our frozen TF++ encoder expects `1024x512`, horizontal FOV `110°`, camera pose `x = -1.5 m, z = 2.0 m` (`src/config/camera.py`).
  - Compatible datasets: **PDM-Lite** (`autonomousvision/PDM_Lite_Carla_LB2`), **SimLingo-Data** (`RenzKa/simlingo`, includes `dreamer/` counterfactual crash labels and `buckets_paths.pkl`; Wayve non-commercial academic license), and **LEAD** (`ln2697/lead`, MIT, 12 towns — verify camera FOV in A46). **Bench2Drive** (`1600x900`, FOV `70°`) and **TaCarla** use incompatible camera geometries and must not feed the frozen backbone directly. **Fail2Drive** (`autonomousvision/fail2drive`, 200 Town13 routes) is for closed-loop evaluation (A58).
  - Arms A–I trained with `crop_bottom_frac=0.0` and `vision_grid=4`; comparable arms must match unless testing crop/grid.
- **PDM-Lite Obstacle & Route Conditioning (S-096 / S-097)**:
  - The 6-town PDM-Lite subset (`Town01–05, Town10`) contains **zero** obstacle scenarios; all obstacle archives are in `Town12` and `Town13` (which make up 69% of Bench2Drive-220).
  - PDM-Lite logs both `route` (expert path shifted around obstacles) and `route_original` (unshifted global plan). Because closed-loop evaluation provides an unshifted global plan, training on obstacle archives requires feeding `route_original` as input/overlay (Arm J, A14) or predicting the shifted path as a head target (A15). Note that Arm J improves obstacle routes (`+7.2 DS`) but regresses on lead-vehicle routes (`-15.2 DS`) due to faster cruising; 80% of vehicle collisions are systematic, and 220 routes resolve ~5 DS at 1 run/arm.

---

## 5. Historical Invalidation Guards & Atari Technical Rules
- **Invalidated Historical Runs (Never Cite Without Qualification)**:
  - *CARLA*: Closed-loop runs before `2026-09-13` (PID controller stuck at constant 20 km/h); `eval_wor_closed_loop.py` runs before `2026-09-14` (wrong camera pose & 2x route lookahead); `bench2drive_agent.py` runs before `2026-09-17` (`args.agent_config` `+` suffix mutation ran an untrained model from route 2 onward, S-022); eval runs missing `frozen_backbone.pth` on `2026-09-25` (S-066/S-069/S-070, S-072).
  - *Atari*: GTrXL runs before commit `049da8e` (`2026-09-17`, policy loss was pure self-imitation with dead critic heads, S-024); off-policy MCTS runs before S-053 (`2026-09-23`, interleaved 8-env replay buffer & weights-only fake resumes).
- **Atari Replay, Mixer & Reanalysis Rules**:
  - Keep `--priority-alpha 0` (uniform replay) unless a validated screen on the S-104 fixed prioritized replay proves otherwise.
  - **Mixer Weight Decay & Attention Collapse**: Under EZ-V2's SGD (`lr=0.2, wd=1e-4`), transformer mixer weights decay by `x0.13` (`e^-2`) every 10k updates to zero by 30–40k (`S-116`, turning GTrXL into a constant bias / ResNet). Setting `--mixer-weight-decay 0` stops decay but leaves SGD mixer weights near init (`S-118`). Naive AdamW `3e-4` (`B37`/`B66`) explodes attention logits (`9k–663k`, one-hot entropy `0.000`) and collapses policy returns. Adding QK-norm (`B67`) keeps attention entropy healthy (~0.90) but still suffered late flat-prior / mixer-norm growth collapse at env ~67k under AdamW `3e-4` — always run `mixer_diag.py` / `mixer_audit.py` and verify non-zero denominators when auditing `|dW|/|W0|`.
  - **Target-Network Arms (B38)**: Before running any pessimistic-search or target-network evaluation (`--pess-lambda`), verify `sd["model"] != sd["target"]` on the checkpoints (B44 checkpoints had `target == model`, making the test a no-op).
  - **Reanalysis Cache & Compilation (B68)**: `--reanalyze-cache`, `--reanalyze-merge` (M1), `--compile-search` (E1), and `--compile-model` (E8) cut `30m` seed runtime from ~34h to ~7.5h (`torch._dynamo.config.cache_size_limit = 64`). Judge numerical equivalence of same-maths/compiled/FP8 changes against an **fp32 reference** (bf16 eager vs fp32 is TV ~0.023), and never promise bitwise-identical weights when batch composition changes (M1 changes batch size, which alters last-bit rounding).

