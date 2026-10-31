---
trigger: always_on
description: CARLA and Atari evaluation protocols, Low vs Epic rendering policy, crash diagnosis rules, TickRuntime classification, and screening rules.
---

# Evaluation Protocol, Crash Diagnosis & Screening Rules

Written after S-137 / S-138 (2026-10-09) and updated through 2026-10-28.

## 1. The Evaluation Protocol is Part of the Result
- **CARLA Rendering Policy (User Policy, 2026-10-14)**:
  - **Low by default, Epic only for routes that segfault at Low.** `carla_garage` training data and Bench2Drive's `leaderboard_evaluator.py` use CARLA's default (`Epic`), whereas our harness runs `-quality-level=Low` (`CARLA_QUALITY`, default `Low`, read by `scripts/eval/run_bench2drive.sh`).
  - All 11 Town03 routes (`24206, 24816, 25378, 25854, 25896, 26393, 26435, 26456, 26950, 26990, 27515`) and a few others (`26956, 26966, 28048`) segfault at `Low` load time but load cleanly at `Epic`.
  - Run those Low-crashing routes as a separate `CKE ...:CARLA_QUALITY=Epic` job for **every** compared arm and report them as their own column (counting records across all relabelled `*_epic*` files `la/lar/larr`).
  - **Never pair a Low run with an Epic run**, or across any differing simulator flag (`-quality-level`, `-vulkan`/`-opengl`, fps, resolution, sensor settings). After every CARLA launch, check `ps` for `-quality-level=` on `CarlaUE4` servers.
- **CARLA Driving Outcomes vs Infrastructure Failures**:
  - **`Failed - TickRuntime` (the leaderboard's 4,000-tick / 200s game-time cap) and `Failed - Agent got blocked` are valid driving outcomes, NEVER infrastructure crashes.** Always include them in DS and paired comparisons, and report the status mix of both sides before quoting a diff.
  - Only `"Agent couldn't be set up"`, `"Simulation crashed"`, and `"Agent crashed"` are infrastructure statuses. Note that `"Agent couldn't be set up"` / `"Agent crashed"` when running 4–6 lanes on a 24 GB GPU is typically a deterministic CUDA OOM at `model.to(device)` (~5.3 GB VRAM per Epic lane); rerun those routes at 2–4 lanes per 24 GB card with verified VRAM headroom (`MIN_FREE_MIB`) after deleting the poisoned JSON records.
- **CARLA Checkpoint Completeness on Eval Boxes**:
  - Frozen-backbone WoR checkpoints save only head weights in `model_epoch_NNN.pth`. Always copy `frozen_backbone.pth` and `run_config.json` with `model_epoch_NNN.pth` and verify `"Merged 494 frozen backbone tensors"` in the guardian log after launch (a missing backbone previously ran random weights and still scored 50–60 DS).
- **Atari Official Evaluation Protocol (2026-10-11 / 2026-10-25)**:
  - **Paper-Protocol Comparison (EZ-V2 400.1 on Breakout)**: **16 simulations**, **60 episodes**, **no sticky actions** (`sticky=0.0`), **no `FireResetEnv`**, `NoopResetEnv(30)`, 27,000-step cap, evaluated on the **final checkpoint** (`--updates 120000`, after the offline phase and LR decay at update ~101,000).
  - **Gotcha**: `eval_ez_checkpoints.py --env-steps 100000` matches *both* `checkpoint_env100000_upd98004.pt` (pre-decay snapshot) and `checkpoint_env100000_upd120000.pt` (final). Always pass `--updates 120000` for official claim numbers, and report snapshot vs final side by side when relevant.
  - **4 Distinct Games per Seed**: Under the deterministic no-sticky protocol, 30 no-op starts yield only **4 distinct games** (one per `noop_count mod 4`). Naive per-episode SE (~7.88) and start-class spread (~36) are not sampling uncertainty; precision comes from **seeds** ($n = \text{seeds}$). Also record the GPU model and batch shape (scores can differ across GPU architectures).
  - **Sticky Actions (`--sticky 0.25`)**: Gives 21–27 distinct games out of 30, used as a **separate robustness column** (and for B2 search claims). Never compare a sticky-action score directly against EZ-V2's 400.1 no-sticky number. In sticky evals, report collapsed no-FIRE seeds (which stall for 27,000 steps at 0 reward) separately; read `max_reward_free_streak` in `eval_loop_stats.py`.

---

## 2. A Route is Not "Broken" Until Shown to Be
Before excluding a route from any run (`a9_exclude.txt`, `gen_lanes.py` CRASH set, lane lists):
1. **Where does it crash?** Check the last non-tick lines of the attempt log: before the first game tick (`> Loading the world`) = map/scenario load; after = driving/agent-dependent (e.g., `23658/23695/23708/24071` completed 42–44 times for plain E15 at DS 100 but crashed under speed switches after `ActorFlow` spawn spam).
2. **Has it ever completed?** Search the `E:\MThesis_EXP` archive (`live_*/**/*.json`) for that route under any agent or flag. One past completion proves the route is not intrinsically broken.
3. **Vary one factor at a time on the same host**: plain checkpoint vs switches, another agent (original WoR), another host, Low vs Epic.
4. **Load-time crash**: Run `scripts/eval/run_town_probe.sh <Town>` with the exact eval server flags (`XDG_RUNTIME_DIR=/tmp`, `-RenderOffScreen -nosound`), then with the suspect flag changed.
5. **Per-Configuration Exclusions**: Exclusions are per `(agent, switches, rendering, host)`. Never copy an `a9_exclude.txt` entry into another arm's route list or the global CRASH set without steps 1–4. Report route coverage (driven / attempted / crash-skipped) with every DS.

---

## 3. Evidence-Based Screening Rules (2026-10-11)
- **CARLA Screening**:
  - Final candidates train **50 epochs**; screens use `--epochs 50 --stop_epoch 20` and resume from `model_epoch_020.pth` to 50. Choose checkpoints by closed-loop DS, never offline validation ADE.
  - The 7 sentinel routes have **zero ranking value** ($\rho = +0.07$, cannot separate a random backbone from trained arms). Use the unselected town-by-weather-stratified **80-route proxy** ($\rho = 0.86$ with 220-route DS; stop rule $> 7$ DS behind reference) for screening, and all **220 routes** (scored via `scripts/analysis/a9_merge.py` with XML-based obstacle classification) for finalists. Single-seed head noise is ~4–8 DS, so require $\ge 2–3$ seeds/runs for claims.
- **Atari Screening**:
  - **No early signal in the archive predicts the 100k score**, and 1,996 extra updates move a 30k score by ~70 points.
  - **Margin-based early stops at 30k are unsafe** (they falsely stop 21–42% of arms that finish level or better). Stop only on **gross failure** ($\ge 3$ seeds below 50% of baseline, or verified attention/projection std collapse).
  - **Exact 30k Resume**: A 30k screen is on the 100k trajectory *only* at `env 30,000 / update 28,004`. Always resume from `checkpoint_env30000_upd28004.pt` + `replay_latest.npz` (resuming from update 30,000 leaves the path because of the offline tail).
