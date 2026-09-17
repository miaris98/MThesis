# Evaluation tiers, and calibrating them against TransFuser++

Companion to [TODO_leaderboard_benchmark.md](../todo/TODO_leaderboard_benchmark.md) (which holds the
published-baseline table this work is ultimately measured against) and
[struggle-solutions.md](struggle-solutions.md) S-011..S-015 (the measurements and failures that
motivated it).

Created 2026-09-15.

---

## Why this exists

The official CARLA Leaderboard 2.0 validation run was measured on 2026-09-15 at **~50 h per
arm** (S-011). That is a perfectly good *final* number and a useless *fitness function*: a
hyperparameter search needs to rank dozens of configurations, and nothing can be searched
against a signal that takes two days per sample.

The response is a three-tier ladder, cheap to expensive. The thing that makes it a ladder rather
than three unrelated scripts is that **every tier drives the agent through the same interface**,
so the same checkpoint - and the same *reference* model - can be run on any of them without
touching agent code.

## The three tiers

| | Tier 1 - fast | Tier 2 - Bench2Drive | Tier 3 - official LB2 |
| --- | --- | --- | --- |
| script | `scripts/eval/run_fast_eval.sh` | `scripts/eval/run_bench2drive.sh` | `scripts/eval/run_leaderboard_official.sh` |
| routes | Town01, random, ~475-650 m | `bench2drive220.xml`, 220 routes | `routes_validation.xml`, 20 routes |
| towns | Town01 (small) | Town12 x104, Town13 x47, + 10 others | Town13 only |
| scenarios | none (ambient traffic only) | Bench2Drive scenario set | ~90 per route, 21 types |
| measured cost | 429 s for 3 routes (`smoke`) | ~4 h on 8x2080ti per carla_garage | ~2.4 h/route, ~50 h for 20 |
| purpose | rank HP configurations | mid-fidelity, published baselines exist | the reported number |

Measured sim-to-wall ratios on the A4000 box: **1.27x** on Town12 (`routes_training.xml`),
**0.216x** on Town13 (`routes_validation.xml`) with two arms contending. The map, not the model,
is the dominant cost - the GPU sat at 42 W of 140 W and 7-30% utilisation while 128 CPU cores
idled at load ~30 (S-015). The evaluator is synchronous lock-step, so throughput is bound by
per-tick round-trip latency and cannot be bought away with a faster GPU.

## The one interface: the Leaderboard `AutonomousAgent` contract

Tiers 2 and 3 already accept **any** agent through `--agent <file.py> --agent-config <string>`,
because both run CARLA's own `leaderboard_evaluator.py`. The contract is:

```
sensors()                          -> list of sensor dicts the harness must spawn
setup(path_to_conf_file)           -> load weights/config
set_global_plan(gps_plan, world_plan)
run_step(input_data, timestamp)    -> carla.VehicleControl
destroy()
```

**Tier 1 is the only tier that does not honour this**, and that is the gap to close.
`scripts/eval/eval_wor_closed_loop.py` currently hard-imports `WorldOnRailsAgent`, spawns a camera itself from
`PDM_LITE_CAMERA` instead of asking `agent.sensors()`, passes a hand-rolled dict
(`{"rgb_front": ..., "speed": ..., "command": ..., "route": ...}`) rather than Leaderboard
`input_data`, and reaches into WoR-specific internals (`agent.net.controller.target_speed`,
`agent.route_points`). A TF++ agent cannot run through it at all.

The fix is not to write a translation layer but to **reuse the leaderboard's own plumbing**,
which is already vendored in `carla_garage/leaderboard/`:

- `leaderboard/autoagents/agent_wrapper.py` -> `AgentWrapper.setup_sensors(vehicle)` spawns
  exactly what `agent.sensors()` declares, and `cleanup()` removes them.
- `leaderboard/envs/sensor_interface.py` -> `SensorInterface.get_data()` returns
  Leaderboard-format `input_data`; `SpeedometerReader` and `CallBack` supply the pseudo-sensors
  (`sensor.speedometer` is not a real CARLA sensor - the leaderboard synthesises it).

Tier 1 then becomes exactly "the leaderboard evaluator, minus the scenario swarm and minus the
huge map", which is the honest description of what a fast proxy should be.

### Sensor asymmetry is handled by the contract, not by special-casing

`setup_sensors()` spawns whatever each agent declares, so:

- **TF++** (`backbone: transFuser` in both pretrained sets) declares camera + IMU + GNSS +
  speedometer + **LiDAR**, and gets all of them.
- **WoR (cnn / qwen30m)** is **camera-only, deliberately and permanently** - no LiDAR, no depth,
  no BEV as *input* (depth/BEV/boxes are supervision targets only). See the standing project
  constraint; this is a thesis scope decision, not an implementation gap. It declares a camera
  and gets a camera.

No branching is needed, and the asymmetry stays visible instead of being buried. **It must be
stated in any comparison**: TF++ is a camera+LiDAR model, our arms are camera-only. TF++'s role
here is a *harness calibration anchor* ("does our pipeline reproduce a known published number?"),
not a like-for-like competitor.

Helpfully, TF++'s camera is `1024x512 @ fov 110`, **identical** to this project's
`PDM_LITE_CAMERA` - both derive from the same PDM-Lite collection geometry - so camera parity
across all three tiers is already correct.

## Town 13 contamination: which pretrained set is legitimate

carla_garage's README is explicit about the validation benchmark:

> *"this is a 'validation' benchmark, so data from Town 13 may not be used during training
> (reminiscent of level 5 driving). To train a model for this benchmark, use the training command
> line option `--setting 13_withheld`."*

Our v3 arms violate this, on two levels:

1. The WoR training set includes Town13 (`/workspace/dataset/wor_trajectories/Town13`;
   `run_label: cnn_v3_full8town`, 8 towns incl. Town12 and Town13).
2. The frozen backbone is TF++'s **`all_towns`** variant
   (`weights_path: .../pretrained_models/all_towns/model_0030_0.pth`), which the README states is
   trained including Town13 and is intended for Bench2Drive and the *test* routes.

So a `routes_validation.xml` number from the current checkpoints would be **contaminated and not
comparable** to TF++'s published validation figures. Both variants are on the box -
`pretrained_models/all_towns` and `pretrained_models/town13_withheld` - so the choice is a
decision, not a blocker:

- **Reporting on Bench2Drive / test routes** -> current `all_towns` training is legitimate, no
  retrain, and `all_towns` TF++ is the correct anchor.
- **Reporting on validation routes (Town13)** -> requires retraining with Town13 excluded *and*
  swapping the frozen backbone to `town13_withheld`.

## The metric: Driving Score is known-broken at these score levels

`driving_score = route_completion x infraction_penalty`, where `infraction_penalty` is a product
over **raw infraction counts** (`0.60 ** n_vehicle_collisions`, etc.), not over per-km rates. RC
grows linearly with distance driven while IP decays exponentially in it, so DS rises and then
falls: **a policy that stops early can outscore a strictly better driver.**

This is not a theory. Two independent confirmations from our own data:

- The 2026-09-13 pilot: qwen beat cnn on *every* per-km infraction rate (vehicle collisions
  6.66 vs 9.26, red lights 0.060 vs 0.297, stop signs 0.720 vs 0.891, blocked 0.180 vs 0.534)
  and drove 1.78x further - and scored **lower** (DS 0.0538 vs 0.0875).
- Our own Tier 1, same route `r000`, same seed, only `max_steps` changed: 600 steps gave
  RC 0.407 / IP 0.420 / **DS 0.171**; 1500 steps gave RC 0.828 / IP 0.064 / **DS 0.053**. It
  drove twice as far and scored a third as well.

carla_garage found the same flaw and published a fix: the **Normalized Driving Score**
(Zimmerlin 2024, Ch. 6), computed by `carla_garage/tools/result_parser.py`. Decisions:

1. Report **NDS** as the primary metric; keep DS for comparability with older published numbers.
2. **Do not search on DS.** Optimising it selects for timid or stalling policies. Search on route
   completion and per-km infraction rates as separate objectives (Optuna supports multi-objective).
3. Hold `max_steps` **fixed** across all trials in a sweep - it is a confound that moves DS
   non-monotonically.

Note our `src/eval/driving_metrics.py` implements **Leaderboard 1.0** (5 infraction types),
deliberately, because World on Rails - the thesis baseline - published on it. Tier 3 runs
**Leaderboard 2.0**, which adds scenario-timeout, yield-to-emergency and minimum-speed penalties.
These are not negligible: the pilot logged 1.11 scenario timeouts/km and 0.42 min-speed
infractions/km for qwen vs 0.119 for cnn. **The proxy and the target therefore differ by
construction, not only by map and scenario density** - a leading candidate explanation if
correlation comes out weak.

## Plan

**Stage 1 - validate the harness against TF++ (no code changes).** Tiers 2 and 3 already take
`--agent`, so point them at `carla_garage/team_code/sensor_agent.py` with
`--agent-config <pretrained_models/{all_towns,town13_withheld}>` (its `setup()` takes a
*directory* containing `config.json`). Compare against the published LB2 technical report
figures. **This gates everything below**: if our harness cannot reproduce a published TF++
number, no proxy calibrated against it means anything. Bench2Drive is the cheaper place to do it.

**Stage 2 - refactor Tier 1 onto the agent contract.** Replace the hand-rolled dict with
`AgentWrapper.setup_sensors()` + `SensorInterface.get_data()` + `agent.run_step(input_data,
timestamp)`. Unifies the two agent classes: `scripts/eval/bench2drive_agent.py` becomes the single agent, with
`WorldOnRailsAgent` as the inner policy it already wraps. Known risks: `AgentWrapper` expects
`CarlaDataProvider` to be initialised, `SpeedometerReader` needs the ego registered, and
`set_global_plan` needs GPS-converted plans.

**Stage 3 - establish the proxy's usable resolution. DONE, and the answer is negative.**
Ran the `hp` preset (12 routes, Town01, `max_steps=1500`, 30 traffic actors, `route_seed=0`) on
v3 cnn and v3 qwen. Because the seed is fixed, both drove **identical** routes - a paired
comparison, the most favourable design available. cnn: 12 routes, 4069 s wall (contended), DS
0.2775 / RC 0.6908 / IP 0.3840, 9 of 12 routes ending in `timeout`.

*Separation - can it rank two models?* **No.**

| paired, 8 common routes | cnn - qwen | SE | distance from 0 |
| --- | --- | --- | --- |
| Driving Score | -0.023 | 0.105 | **0.22 SE** |
| Route Completion | -0.014 | 0.132 | **0.10 SE** |

Roughly 2 SE is needed to claim a difference; this is 0.22. Pairing did **not** rescue it,
because the two models fail on *different* routes - route difficulty is not a shared latent
factor that cancels in the difference. Resolving the observed ~0.08 DS gap at SE 0.02 would need
**~220 routes, about 18 h per model**, which is no longer a fast eval.

*Which metric is least noisy?* Measured, not assumed:

| run | metric | mean | sd | CV = sd/mean |
| --- | --- | --- | --- | --- |
| cnn | DS | 0.2625 | 0.2770 | **1.06** |
| cnn | RC | 0.6962 | 0.2719 | **0.39** |
| qwen | DS | 0.3446 | 0.1369 | 0.40 |
| qwen | RC | 0.7161 | 0.2778 | **0.39** |

cnn's Driving Score has a **standard deviation larger than its mean**. Route Completion sits at
CV 0.39 for *both* models, because it is bounded in [0,1] while DS multiplies it by a penalty
exponential in infraction count, which is heavy-tailed by construction. **RC is ~2.7x less noisy
than DS** - the "do not search on DS" argument above is now measured, not just principled.

Also: qwen (0.345) scored *above* cnn (0.263) here, while the official pilot had cnn above qwen.
Opposite ordering. Neither result is significant, so this is not evidence the proxy is wrong -
but it is emphatically not evidence that it is right.

**Consequence: Tier 1 is a screening tool, not a ranking tool.** It will reject a
catastrophically broken configuration - a bad learning rate, a broken loss weight - which is most
of what an early search needs to discard. It will not order two decent candidates and must never
be used to pick a winner. Anything surviving the screen is promoted to Tier 2.

Before spending routes to buy resolution, test variance reduction: **9 of 12 cnn routes ended in
`timeout`**, i.e. episodes cut off mid-drive at `max_steps`, which injects variance directly into
both RC and IP. Letting routes actually finish may buy more than more routes would.

**Stage 4 - rank correlation.** Ladder of ~10 checkpoints spanning capability (full8town epochs
005-050 and v3 epochs 005-031 exist for both arms, plus TF++ as a high anchor). Reference tier =
Tier 2, or Tier 3 restricted to a few routes. Spearman rho; want >= ~0.7 before trusting it to
drive a search. Caveat: an epoch ladder only proves the proxy tracks *training progress* - the
ladder must span architectures (it does: cnn + qwen) because a search varies LR, architecture and
loss weights.

## Stage 2: DONE - Tier 1 refactored onto the real agent contract

`scripts/eval/eval_wor_closed_loop.py` no longer spawns its own camera or hand-rolls a policy-input dict.
`run_route()` now builds a FRESH agent per route via `load_agent_class(--agent)` +
`agent_cls(host, port)` + `agent.setup(agent_config)`, converts its own `route_locations` into a
Leaderboard-format plan (`location_route_to_gps` + `RoadOption.LANEFOLLOW`, matching
`leaderboard_evaluator.py`'s own mechanism) and calls `agent.set_global_plan(...)`, spawns
whatever `agent.sensors()` declares via `AgentWrapper.setup_sensors(ego)`, and drives by calling
`agent_wrapper()` each tick - exactly `AutonomousAgent.__call__()`, which pulls sensor data via
`SensorInterface.get_data(GameTime.get_frame())` and prints the same `Ratio = ...x` line Tiers
2/3 already print. `scripts/eval/run_fast_eval.sh` gained the same `EVAL_AGENT` / `EVAL_AGENT_CONFIG` /
`EVAL_PYTHON` triple as Tier 3.

**Regression-validated against our own model before trusting it for anything else.** Same
checkpoint (v3 cnn), same `smoke` preset (3 routes, seed 0), old hand-rolled path vs new
contract-based path:

| | old (hand-rolled camera+dict) | new (real AgentWrapper contract) |
| --- | --- | --- |
| Driving Score | 0.309 | 0.300 |
| Route Completion | 0.542 | 0.542 (identical) |
| route r002 | DS=0.700 RC=1.000 IP=0.700 completed | **identical** |

Within noise of itself - the refactor preserves driving/scoring behaviour while now running
through code the Leaderboard evaluator itself uses, not a lookalike.

**Four real bugs found by actually running it, not by review:**
1. `scripts/eval/run_fast_eval.sh`'s `PYTHONPATH` never included `leaderboard`/`scenario_runner` (Tier 1 never
   needed them under the old hand-rolled path) - `ModuleNotFoundError: No module named
   'leaderboard'` on the very first import.
2. `location_route_to_gps` (leaderboard's own vendored helper) requires `carla.Transform`
   objects (`transform.location`), not bare `carla.Location` - `route_locations` only ever held
   Locations. Fixed by wrapping: `carla.Transform(loc)`.
3. `CarlaDataProvider.get_hero_actor()` - which `WorB2DAgent._get_hero()` calls - searches
   `_carla_actor_pool`, a THIRD dict distinct from the one `register_actor()` populates
   (`_actor_velocity_map`/`_actor_location_map`). Only `request_new_actor()` populates the pool
   normally; this function spawns the ego directly via `world.try_spawn_actor()` for exact
   route-origin placement, so nothing else ever added it. Fixed by registering into the pool
   manually, and explicitly removing it on cleanup (`get_hero_actor()` returns the FIRST
   dict-order match, so a stale entry from route N-1 would shadow route N's hero forever if left
   in place).
4. **The subtle one**: `GameTime.restart()` called AFTER `AgentWrapper.setup_sensors()` instead
   of before it. GNSS/IMU/speedometer are pseudo-sensors
   (`leaderboard.envs.sensor_interface.BaseReader`) backed by a background thread that captures
   `latest_time = GameTime.get_time()` ONCE, at thread-start (inside `setup_sensors()`), and only
   fires again once `GameTime` advances past that captured value. Route 1 worked (GameTime was
   already at 0 from process start). Route 2 reproducibly failed with
   `SensorReceivedNoData: A sensor took too long to send their data`: the pseudo-sensor thread
   captured route 1's high ending game-time as `latest_time`, then the clock was reset to 0
   immediately after, so `current_time - latest_time` stayed negative for nearly the whole route.
   Fixed by moving `GameTime.restart()` to before agent construction/sensor setup, so every
   route's pseudo-sensor threads start against an already-reset clock, matching route 1's
   (accidentally correct) behaviour exactly.

**TF++ ran end-to-end through Tier 1** immediately after, no further code changes needed - proof
the contract is genuinely agent-agnostic, not agent-agnostic-for-WoR-only. `smoke` preset (3
routes, Town01, seed 0), `town13_withheld` weights: DS 0.1275, RC 0.2054, IP 0.6742, all 3 routes
`timeout` (none reached the destination within 600 steps / 30s). **Wall time: 775 s for 3
routes** (~13 min) - against the ~6-12 h estimated for TF++ on `routes_devtest.xml`'s full-length
Town12 routes. This is the entire point of Stage 2: TF++ went from impractical to a routine
calibration check.

**Do not read the DS numbers above as a TF++-vs-ours comparison yet.** 3 routes is far below the
~20 Stage 3 found necessary to resolve a real difference between two policies, and TF++'s three
`timeout` terminations at a 30 s cap plausibly reflect a careful/slower driving style meeting a
short cap tuned for our own models, not a quality signal - IP=0.674 (clean driving, not crashing)
is consistent with that reading, but 3 routes cannot distinguish it from TF++ genuinely
underperforming here. Re-run at the `hp` preset (more routes, larger `max_steps`) before drawing
any conclusion.

## Stage 1, revised: Town13-validation is not a credible calibration target - Bench2Drive is

Stage 1's original plan (line 154) was to compare our harness's TF++ score against "the published
LB2 technical report figures" on `routes_validation.xml`/Town13. That plan is retired as of
2026-09-15: **the public checkpoint's own Town13-validation performance is independently
documented as low and noisy**, so a mismatch there would not tell us anything about our harness.

### What triggered this: diagnosing why Tier 1's TF++ smoke score looked bad

The user asked whether the 13-minute Tier 1 proxy "is actually robust... does the tf++ provide
similar results to the official leaderboard". Investigating meant adding real diagnostics rather
than guessing - `carla_garage/team_code/sensor_agent.py`'s stuck/creep-recovery logging was
extended (tracked as `patches/carla_garage_sensor_agent_diagnostics.patch`, since `Carla-utils/`
is gitignored and the vendored clone is its own nested git repo; auto-applied by
`scripts/setup/setup_tfpp_env.sh`) to print, on every "Creeping stopped by safety box" tick: the LiDAR point
count inside the box, the box's near-boundary coordinates, and the nearest tracked
vehicle/walker actor + distance. `scripts/eval/eval_wor_closed_loop.py` now also prints every counted
collision/red-light infraction live (route_idx, timestamp, actor), not only at the end in the
JSON.

**Town01 smoke test (3 routes, `max_steps=3000`):**
- r000 **permanently deadlocked** in TF++'s own creep-recovery safety-box check: 981 consecutive
  "Creeping stopped" events, 0 successful recoveries, over the full 150 s cap. The LiDAR points
  filling the box sat at x = 2.4508-2.4513 m - matching `config.ego_extent_x =
  2.4508416652679443` to 5 significant figures, i.e. the box's own near boundary - while the
  nearest tracked actor was 10-15 m away and receding, eventually "none within 15 m". A **self-hit
  signature**, not a real obstruction.
- r001: 3 vehicle collisions + 5 red-light violations + 1 static collision, in just 410 m driven.
  Zero safety-box/stop-sign involvement - a genuine driving failure, unrelated to r000's deadlock.
- r002: DS = 1.0, clean.
- Aggregate: **16.9 infractions/km** over 0.71 km driven.

**Town12 smoke test (same config, TF++'s actual training town; route manifest mean length 4963 m
- so RC is capped by the 150 s time budget on this huge map regardless of driving quality, and
DS is not comparable to Town01's on that axis):**
- r000 `blocked`, but on a **different, legitimate** mechanism: `nearest_actor=
  vehicle.mitsubishi.fusorosa at 7.18m`, consistently, tick after tick - a real bus, physically
  plausible geometry for a ~12 m vehicle to have LiDAR-visible surface points closer than its
  reported center distance. This followed an actual collision with that same bus at t=65 s: TF++
  crashed into it, then correctly refused to creep forward into it. Working as intended, not a
  bug - and structurally unlike r000 on Town01.
- r001, r002: clean - one red-light violation each, zero collisions, zero safety-box events.
- Aggregate: **1.81 infractions/km** over 2.21 km driven - **~9x lower** than Town01.
- The self-hit signature never recurred once on Town12's 3 routes.

This is consistent with distribution shift (`routes_training.xml`/`routes_validation.xml`
confirm TF++ never sees Town01 - only Town12 and Town13), but before spending more routes chasing
that theory, the more decisive check was to look at what TF++'s own community reports.

### External corroboration: the public checkpoint is known to be low-scoring and noisy on Town13

- **[carla_garage issue #120](https://github.com/autonomousvision/carla_garage/issues/120)**
  (opened 2026-08-10, still open, no maintainer response): an independent reproduction using the
  **exact same checkpoint** we run (`town13_withheld/model_0030_0`), on the actual
  `routes_validation_split` benchmark, via the standard vanilla evaluator on CARLA 0.9.15. Result:
  "average driving score was below 1, with an average route completion of roughly 37%", most runs
  ending "Agent got blocked" - collisions with dynamic objects "despite apparent detection" and
  difficulty recovering when stuck behind an obstacle. The same symptom cluster we found,
  independently, on TF++'s own validation town.
- **Hidden Biases of End-to-End Driving Datasets** (arXiv 2412.09602, Table 5): TF++ on Town13
  validation reports **RC 50.2%, Infraction Score 0.10, DS 1.08** (of 100) - and the paper notes
  "identical agents yielding results that differ by more than 1 DS", i.e. the benchmark is
  documented as highly noisy run-to-run even for a fixed checkpoint.

**Conclusion:** `routes_validation.xml`/Town13 is not a usable calibration target for *anyone*
running the public checkpoint right now, including us. A mismatch there indicts the benchmark's
known instability before it indicts our harness. This is a stronger and more useful conclusion
than "Town01 is out of distribution" - it means chasing exact agreement on Town13 validation was
never going to succeed regardless of which town Tier 1 defaults to.

### The credible replacement: Bench2Drive's published table

Bench2Drive ([NeurIPS 2024 D&B track](https://proceedings.neurips.cc/paper_files/paper/2024/file/017761f94a1cd66d01c041aff85492c4-Paper-Datasets_and_Benchmarks_Track.pdf))
is cross-validated by many independent follow-up papers reporting on the same 220-route table,
which Town13-validation is not:

| model | Driving Score | Success Rate |
| --- | --- | --- |
| AD-MLP | 18.05 | 0.00% |
| TCP | 40.70 | 15.00% |
| VAD | 42.35 | 15.00% |
| UniAD | 45.81 | 16.36% |
| ThinkTwice | 62.44 | 31.23% |
| DriveAdapter | 64.22 | 33.08% |
| **TF++ (Baseline)** | **84.21** | **67.27%** |
| TF++ w/ VLAAD-MIL (SOTA) | 86.97 | 71.97% |
| **PDM-Lite (Expert)** | **97.02** | **92.27%** |

Two reasons this is structurally more usable than Town13 validation, not just better-documented:

1. Bench2Drive's 220 routes are **short (~150 m) with exactly one scripted scenario each**,
   distributed across all CARLA towns. That is the same order of magnitude as Tier 1's own
   Town01 routes, and it starves the failure mode found above: the creep-recovery deadlock needs
   ~55 s of continuous near-zero velocity (`stuck_threshold=1100` ticks) before it can even
   engage, which a ~150 m route very often finishes or fails before reaching.
2. **PDM-Lite is a privileged, rule-based expert** (not a learned policy), already vendored at
   `carla_garage/team_code/autopilot.py` and `Bench2Drive/leaderboard/team_code/autopilot.py`. Its
   published 97.02 DS / 92.27% SR is a near-ceiling anchor that isolates a different question than
   TF++ does: does **our scoring/harness plumbing** reproduce a near-perfect number, independent
   of any learned model's generalization noise? TF++ tests "does a learned reference model
   transfer through our pipeline"; PDM-Lite tests "is the pipeline itself correct". Both are
   useful and they fail in different ways if something is wrong.

**Practical gap:** `scripts/eval/run_bench2drive.sh` (Tier 2) does not yet have the `EVAL_AGENT` /
`EVAL_AGENT_CONFIG` / `EVAL_PYTHON` pluggable-agent mechanism that `scripts/eval/run_fast_eval.sh` and
`scripts/eval/run_leaderboard_official.sh` already received this session - it is still hardcoded to
`scripts/eval/bench2drive_agent.py`. That is the next piece of work before TF++ or PDM-Lite can actually run
through Tier 2.

**Revised Stage 1:** compare our harness's TF++ and PDM-Lite scores against the Bench2Drive table
above (84.21 DS / 67.27% SR for TF++; 97.02 DS / 92.27% SR for PDM-Lite), not against
`routes_validation.xml`. `routes_training.xml`/Town12 remains useful for Tier 3 as the *training*
protocol check (no withheld-town concern), but Town13 validation is retired as a target until (if
ever) the upstream noise issue in #120 is resolved.

## Status

**Established (measured):**
- Tier 1 at 12 routes cannot separate v3 cnn from v3 qwen - 0.22 SE on a paired comparison
  (Stage 3). It is a screen, not a ranker.
- RC is ~2.7x less noisy than DS and is the better search objective.
- The DS metric flaw is real and reproduces in our own harness.
- Stage 1 *plumbing* works: TF++ runs through `scripts/eval/run_leaderboard_official.sh` unchanged, with its
  own venv and its own sensor set including LiDAR.

**Measured cost of TF++ as a reference:** it drives at a **0.040x** sim-to-wall ratio on Town12
(775 s of wall time bought 32.8 s of simulation), against 1.27x for our camera-only WoR arms on
the same town - roughly 30x more expensive per tick, from LiDAR raycasting plus a much larger
model. TF++ is therefore viable as a **one-off calibration anchor** and not as a routine
reference; budget whole routes, not whole benchmarks.

**Explicitly not yet established:**
- That Tier 1 correlates with Tier 2 or Tier 3 at all.
- That Tier 2 correlates with Tier 3 (Stage 4's reference-tier substitution assumes it).
- That our harness reproduces any *published* TF++ number - only that TF++ runs in it, on both
  Tier 1 (cheap, validated end-to-end) and Tier 2/3 (expensive, plumbing-only validated - the
  devtest run was killed before finishing for being too slow, never reached a score). The target
  for this comparison changed 2026-09-15 (see "Stage 1, revised" above): `routes_validation.xml`
  is retired as a calibration target (independently documented as low-scoring and noisy for this
  exact checkpoint, not just by us), replaced by Bench2Drive's published table
  (TF++ 84.21 DS / 67.27% SR, PDM-Lite 97.02 DS / 92.27% SR). `scripts/eval/run_bench2drive.sh` still needs
  the `EVAL_AGENT` pluggability fix before this comparison can actually be run.
- Any TF++-vs-ours ranking. The one data point that exists (Stage 2's TF++ smoke run) is 3
  routes, an order of magnitude below Stage 3's ~20-route resolving threshold, and must not be
  treated as a comparison.
- Whether the Town01 self-hit deadlock (found 2026-09-15) recurs systematically or was one unlucky
  spawn - not re-tested after the Town12 run showed a structurally different `blocked` cause
  (a real post-collision obstruction). Lower priority now that Town13-validation-style long routes
  are no longer the calibration target; Bench2Drive's short routes are far less exposed to it.

Until that last point passes, `scripts/eval/run_fast_eval.sh` output is a development signal only and must not
appear in the thesis as a driving score.
