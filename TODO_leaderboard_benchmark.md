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

## Option A tried, hit a real wall: CARLA 0.9.10.1 is gone

2026-09-09: before writing any code, checked whether 0.9.10.1 — the version the leaderboard
docs insist on, "the exact version used by the online servers" — is still obtainable.
It isn't, through any channel found:

- The official CDN link (`tiny.carla.org/carla-0-9-10-1-linux` →
  `carla-releases.b-cdn.net/Linux/CARLA_0.9.10.1.tar.gz`) returns **403 Forbidden** — and this
  reproduces from a plain local `curl`, not just from vast.ai, so it is not an IP/datacenter
  block. CARLA appears to have stopped serving this specific legacy release.
- The community SourceForge mirror (`carla-simulator.mirror`) only goes back to **0.9.13** —
  nothing for 0.9.10.x.
- The GitHub release page for `0.9.10.1` lists no alternate mirror (no Google Drive/Baidu/
  etc., unlike TCP's own dataset links).

Decision (user, 2026-09-09): drop the second-CARLA-version plan rather than go looking for an
unofficial mirror of a multi-GB precompiled Unreal Engine binary — that is a real
supply-chain trust call, not something to route around unilaterally. **Building Option B
instead.** Option A's prerequisite repos are already cloned and left in place in case
0.9.10.1 resurfaces somewhere trustworthy later: `/workspace/leaderboard` (branch
`leaderboard-1.0`) and `/workspace/scenario_runner` (branch `0.9.10`) — these clones are what
Option B's route files are actually read from (see below), so they were not wasted effort.

## Option B — status: built, mid-validation

Reuses the current CARLA 0.9.15 install and `eval_wor_closed_loop.py`'s existing agent loop.
Pulls in the official route pool; skips `scenario_runner` entirely (no scripted dynamic
scenarios — ambient traffic only, same as every run so far).

**Done:**
- `eval_wor_closed_loop.py`: added `load_official_routes(xml_path, town, num_routes, seed)`,
  which parses a Leaderboard 1.0 `routes_*.xml`, filters by town, and deterministically
  subsamples if there are more routes than requested. New CLI flags `--route_source
  {random,official}` (default `random`, so every prior invocation of this script is
  unaffected) and `--route_file`.
- `run_route()` now branches: an official-route spec carries `waypoints_raw` (a list of
  `(x, y)` pairs, not the XML's raw z/pitch/yaw/roll — some shipped entries have values like
  `pitch="360.0"`, not safe to trust directly). Each point is snapped to the nearest driving
  lane via `get_waypoint(project_to_road=True)`, then `GlobalRoutePlanner.trace_route()` is
  chained between every consecutive pair of snapped points, so the vehicle actually follows
  the route's full shape (routes carry up to ~29 control points) rather than a single
  start/end trace that would let the planner pick its own path and silently discard it.
- Confirmed the route files themselves: `/workspace/leaderboard/data/routes_training.xml`
  (50 routes: Town01×10, Town03×20, Town04×10, Town06×10) and `routes_testing.xml` (26
  routes: Town02×6, Town04×10, Town05×10). No `<weathers>` block in either file — weather
  isn't in the XML, the real leaderboard evaluator assigns it separately, so this still needs
  a decision before a real run (see Open below). There's also a separate
  `all_towns_traffic_scenarios_public.json` (the scripted-scenario trigger definitions) which
  Option B does not use, by design — that's the scenario_runner-shaped fidelity gap already
  named above.
- Town coverage check against the existing CARLA 0.9.15 install: Town01–05 and Town10HD were
  already present; **Town06 was missing** (only an `.xodr` road-network file, no compiled
  `.umap`). Fetched `AdditionalMaps_0.9.15.tar.gz` (official, still live —
  `downloads.carlasim.com`, unlike 0.9.10.1's dead link) and ran `ImportAssets.sh` to add it.

- [x] `ImportAssets.sh` completed and Town06 verified loadable
      (`client.load_world("Town06")` + `world.tick()`, 436 spawn points, no error). The tar
      output did include `Unexpected inconsistency when making directory` warnings on a
      handful of generic `Engine/Content/` editor-support files (not CARLA map content); did
      not investigate further since the actual map load works.
- [x] **Smoke-tested `--route_source official` end to end — works.** 3 routes on Town04 via
      `routes_testing.xml`, clean run, correct route IDs (`official_r12`, `official_r15`,
      `official_r6`, matching the deterministic subsample). Route completion was low (7-10%,
      all `timeout`) only because the smoke test used a deliberately short `--max_steps 800`
      (40 s) against routes whose mean length (~1273 m estimated) is far longer than this
      project's own random routes (100-500 m) — a real run needs a much larger `--max_steps`,
      not a code fix.

**Found and resolved during validation: a real crash, but not in the new code.**
`--route_source official` on **Town05** crashed reproducibly — always at the same point (a
CARLA-server-side segfault, `Signal 11`, right after checkpoint loading, before the first
route's first tick). Isolated with two scratch scripts run against a live server: route
loading + snap-to-lane + chained `trace_route()` alone (no torch) completed cleanly on
Town05, and ego+sensor spawn alone (no torch) also completed cleanly on Town05 — so the new
route code was not the trigger. Then the **pre-existing, long-proven-stable `--route_source
random` path** (unchanged code, used for every prior closed-loop run in this project) was
run on Town05 and **crashed identically, at the identical point**. Conclusive: this is a
pre-existing environment instability specific to Town05, never discovered before because
every closed-loop run in this project's history used Town01 exclusively until this session.
Town04 (also genuinely held-out) was tried next and works cleanly with both `random` and
`official` route sources — the instability is Town05-specific, not "any non-Town01 town."
Not investigated further (likely VRAM/scene-complexity related — Town05 is a larger map than
Town01/04 — but unconfirmed); **avoid Town05 for closed-loop runs on this instance class
until someone diagnoses it properly.**

**Resolved, before the real run:**
- [x] **Weather.** Cycled through 14 standard CARLA presets by route index
      (`WEATHER_PRESETS[(route_index - 1) % 14]`) rather than leaving the server default —
      closer to the spirit of the leaderboard's per-route weather variation, though not its
      actual (non-public) assignment.
- [x] **Routes.** Town04 only, 10 routes from `routes_testing.xml`, `--route_seed 0`. Town05
      stayed off the table (crash finding above). All 3 arms driven on byte-identical routes
      and traffic (`--num_vehicles 20`), which is what makes the paired comparison below valid.
- [x] **`--max_steps`.** Set to 8000 (400s sim time) against the ~1273m mean route length;
      `--blocked_timeout_s 45`.

## Result — Town04, official routes, all 3 arms (2026-09-09)

`run_closed_loop_official.sh`, 3 concurrent CARLA servers (ports 2000/2010/2020), same
checkpoints as 13.31 (seed 0).

| Arm | Driving Score | Route Completion | Infraction Penalty | Terminations |
|---|---|---|---|---|
| cnn | 0.304 | 0.594 | 0.571 | 10 timeout |
| qwen30m (baseline) | 0.257 | 0.548 | 0.517 | 9 timeout, 1 blocked |
| qwen30m+geom | 0.274 | 0.602 | 0.533 | 10 timeout |

Paired bootstrap (`compare_closed_loop.py`, `driving_score`, 10 shared routes) — **all three
pairings NOT SUPPORTED**:

| Pairing | A − B | 95% CI |
|---|---|---|
| cnn vs qwen30m | +0.0477 | [−0.0002, +0.1161] |
| cnn vs qwen30m+geom | +0.0309 | [−0.0003, +0.0918] |
| qwen30m vs qwen30m+geom | −0.0168 | [−0.0617, +0.0102] |

None of the three CIs exclude zero — at n=10 nothing here is distinguishable from route-draw
noise, so no ranking claim is supportable from this run alone. The notable point is *direction*:
13.31 (Town01, in-distribution, n=15) found cnn vs qwen30m+geom was the one pairing that
**did** clear the bootstrap, with qwen30m+geom ahead (CI [−0.148, −0.012] in RC terms — see
13.31). Here, out-of-distribution, the point estimate for that same pairing flips — cnn nominally
ahead by +0.031 — though the CI ([−0.0003, +0.0918]) is the closest of the three to excluding
zero in cnn's favor, not geom's. Read together: **the geometry-loss advantage found
in-distribution does not clearly replicate out-of-distribution**, and a second training seed
(seed 1, running now — see below) is needed before treating even that directional flip as
anything more than one more coin flip (the exact 13.29 lesson: same-signed point estimates
without a cleared CI are not evidence).

All three arms terminate almost exclusively by `timeout` rather than collision/off-road — Town04
is a highway-and-mountain map genuinely unlike the flat suburban grid (Town01/02/03/10) all
three policies trained on, so low completion here reads as "doesn't know this road geometry"
more than "drives badly." Consistent with the pre-registered expectation in this doc's
"honest ceiling" section below.

## Either way — the honest ceiling

Even a fully successful run only reaches "closest legitimate proxy available," not the
literal routes behind the 31.4/69.7/75.1 numbers above. State it that way in any writeup that
cites this comparison — don't let a hard-won number quietly imply more than it earned.

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
