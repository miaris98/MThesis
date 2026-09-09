# TODO — Official CARLA Leaderboard protocol, and TCP

Companion to [TODO_wor_architecture.md](TODO_wor_architecture.md) and
`challenges/challenges_13_transformer_head_underperformance.md` (13.31, the closed-loop
result this follows on from).

Created 2026-09-09.

---

## Why this exists

13.31's closed-loop numbers (cnn 0.244, qwen30m 0.278, qwen30m+geometry 0.327, on a 0-1
scale) are not on the same footing as the published numbers for the methods this thesis is
positioned against:

| Model | Driving Score (0-100) | Route Completion | Infraction Penalty | Source |
|---|---|---|---|---|
| World on Rails | 31.4 | 57.7 | 0.56 | official leaderboard, published in the WoR paper |
| TCP | 69.7 | 83.0 | 0.85 | official leaderboard, published in the TCP paper |
| TCP-Ens | 75.1 | 85.6 | 0.87 | official leaderboard, published in the TCP paper |
| cnn (ours) | 24.4 | 87.1 | 0.29 | this project, Town01 only, 15 routes |
| qwen30m (ours) | 27.8 | 90.5 | 0.31 | this project, Town01 only, 15 routes |
| qwen30m+geom (ours) | 32.7 | 90.1 | 0.36 | this project, Town01 only, 15 routes |

The published numbers were scored by CARLA's own online evaluation server against **secret**
held-out routes across held-out towns, varied weather, and scripted dynamic scenarios (a car
running a red light, a pedestrian emerging, etc.). Our numbers are 15 routes in **Town01 — a
town in our own training data** — with plain ambient traffic and no scripted scenarios. It is
a materially easier test, so "our 32.7 beats WoR's published 31.4" is not a valid claim as it
stands.

**Hard limit: Leaderboard 1.0 is deprecated.** The exact secret routes that produced the
numbers above are gone — there is no live server left to submit to for a fresh, literally
identical-protocol number. The best available fix is the closest legitimate *proxy*: the
leaderboard's own public route pool, which the original papers themselves used for local
validation before their real submission. This gets us close, not identical.

## Two build options — keep both on the table

Per 2026-09-09 discussion: don't commit to one path yet, scope both so a future session can
pick based on time available.

### Option A — proper build (closest legitimate number)

- Second CARLA install, **0.9.10.1** specifically — the leaderboard docs are explicit that
  this is "the exact version used by the online servers," pinned, not a suggestion. This
  project's existing instance runs 0.9.15 for everything else; this would sit alongside it,
  not replace it.
- Clone `carla-simulator/leaderboard` and `carla-simulator/scenario_runner`, wire up their
  env vars. `scenario_runner` is a hard dependency for the actual triggered scenarios along
  each route — without it this degrades to Option B.
- Write a new agent class implementing the five-method `AutonomousAgent` interface
  (`get_entry_point`, `setup`, `sensors`, `run_step`, `destroy`) that wraps the existing
  `wor_policy.py` / `qwen_wor_policy.py` inference — this is a new adapter, not a reuse of
  `eval_wor_closed_loop.py`'s current agent loop, since the interface shape is different.
- Run on `routes_training.xml` (50 routes, 112.8 km) and/or `routes_testing.xml` (26 routes,
  58.9 km) from the leaderboard repo's `data/` folder — the 8 public towns, official weather
  presets.
- **Risk:** a second CARLA version on the same box means redoing the Vulkan/userspace-driver
  matching dance from challenges_01 §1.1-1.6a for a *different* CARLA build. That saga was
  expensive the first time; budget real time for it to happen again.
- **Payoff:** this is also most of what's needed to run TCP itself (see below) — TCP already
  targets 0.9.10.1, so Option A's CARLA install and scenario_runner setup remove two of TCP's
  three blockers for free if that thread gets picked up too.

### Option B — lighter, unverified middle ground

- Keep the current CARLA 0.9.15 install and `eval_wor_closed_loop.py`'s existing agent loop.
- Pull in just the official town list, route waypoints, and weather presets from
  `routes_training.xml`/`routes_testing.xml`, and feed them into `build_route_manifest()` in
  place of the current self-generated routes.
- Skip `scenario_runner` — no scripted dynamic scenarios, ambient traffic only, same as now.
- **Not yet verified feasible.** Two open questions before committing time: (1) does the
  leaderboard's route XML format parse cleanly against a 0.9.15 map, given maps can change
  between CARLA versions in ways that shift spawn points or invalidate route waypoints; (2)
  does skipping scenario_runner's scenarios undermine the comparison enough that it isn't
  worth doing at all — the published numbers include scripted-scenario infractions that this
  option would structurally be unable to reproduce. Check both before starting, not after.
- **Payoff:** no second CARLA install, reuses all existing infra, much smaller time cost than
  Option A if it turns out to work.

### Either way — the honest ceiling

Even a fully successful Option A run only reaches "closest legitimate proxy available," not
the literal routes behind the 31.4/69.7/75.1 numbers above. State it that way in any writeup
that cites this comparison — don't let a hard-won number quietly imply more than it earned.

## TCP, separately

The original ask this session was "test TCP instead of WoR." That's a larger, mostly
independent thread from the leaderboard-protocol question above, blocked on different things:

- **No official pretrained checkpoint.** Only a community reproduction on Hugging Face
  (`maxim-igenbergs/tcp-carla-repro`), unverified provenance and training fidelity. Usable
  for a rough smoke test, not a controlled comparison.
- **Training from scratch** needs either TCP's own ~115GB custom-format expert dataset
  (Huggingface/GoogleDrive/BaiduYun per its repo — user flagged willingness to try pulling
  this) or a from-scratch adapter converting our PDM-Lite data into TCP's expected input
  format (BEV + measurements shape, not the WoR waypoint-target format) — nontrivial, TCP's
  format was never designed against PDM-Lite.
- **Different output paradigm.** TCP predicts a trajectory *and* direct control
  (steer/throttle/brake) through a fusion module — it does not hand waypoints to a PID
  controller like every model in this project so far. It needs its own agent wrapper
  regardless of which leaderboard option above gets built; it cannot reuse
  `wor_policy.py`/`qwen_wor_policy.py`'s `act()` shape at all.
- Same CARLA-version note as Option A: TCP targets 0.9.10.1, so building Option A first
  de-risks this thread too.

## Sources

- [WoR paper (arXiv)](https://arxiv.org/html/2105.00636v1) — official leaderboard numbers, Table 1
- [TCP paper (arXiv)](https://arxiv.org/pdf/2206.08129) — official leaderboard numbers
- [TCP GitHub](https://github.com/OpenDriveLab/TCP) — CARLA 0.9.10.1 requirement, dataset links
- [TCP reproduced checkpoint (Hugging Face)](https://huggingface.co/maxim-igenbergs/tcp-carla-repro)
- [CARLA Leaderboard 1.0 — Get Started](https://leaderboard.carla.org/get_started_v1_0/) — deprecated notice, setup steps, `AutonomousAgent` interface
- [CARLA Leaderboard 1.0 — Evaluation criteria](https://leaderboard.carla.org/evaluation_v1_0/)
