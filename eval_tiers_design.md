# Evaluation tiers, and calibrating them against TransFuser++

Companion to [TODO_leaderboard_benchmark.md](TODO_leaderboard_benchmark.md) (which holds the
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
| script | `run_fast_eval.sh` | `run_bench2drive.sh` | `run_leaderboard_official.sh` |
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
`eval_wor_closed_loop.py` currently hard-imports `WorldOnRailsAgent`, spawns a camera itself from
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
timestamp)`. Unifies the two agent classes: `bench2drive_agent.py` becomes the single agent, with
`WorldOnRailsAgent` as the inner policy it already wraps. Known risks: `AgentWrapper` expects
`CarlaDataProvider` to be initialised, `SpeedometerReader` needs the ego registered, and
`set_global_plan` needs GPS-converted plans.

**Stage 3 - establish the proxy's usable resolution.** Before any correlation claim:
- *Repeatability*: same checkpoint, several `FAST_EVAL_SEED` values -> score variance. carla_garage
  warns evaluation variance "is quite high" and recommends 3 seed repetitions even at Tier 3.
  Early Tier 1 data shows per-route DS spanning 0.05-0.42 for one model, so this is the binding
  constraint on how many routes a preset needs.
- *Separation*: does Tier 1 reproduce the pilot's cnn > qwen DS ordering, in the specific case
  where DS disagrees with "drives better"? That is the sharpest single test available.

**Stage 4 - rank correlation.** Ladder of ~10 checkpoints spanning capability (full8town epochs
005-050 and v3 epochs 005-031 exist for both arms, plus TF++ as a high anchor). Reference tier =
Tier 2, or Tier 3 restricted to a few routes. Spearman rho; want >= ~0.7 before trusting it to
drive a search. Caveat: an epoch ladder only proves the proxy tracks *training progress* - the
ladder must span architectures (it does: cnn + qwen) because a search varies LR, architecture and
loss weights.

## What is explicitly not yet established

- That Tier 1 correlates with Tier 2 or Tier 3 at all. **Nothing** has been validated yet.
- That Tier 2 correlates with Tier 3 (Stage 4's reference-tier substitution assumes it).
- That our harness reproduces any published number (Stage 1).

Until Stage 1 passes, `run_fast_eval.sh` output is a development signal only and must not appear
in the thesis as a driving score.
