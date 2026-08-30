# CARLA PPO Driving Policy & Exploration Improvement Roadmap

## 1. Executive Summary & Problem Diagnosis
During RL training, the agent fell into a passive local optimum: **holding throttle at ~1% while steering slightly right**.
* **The Root Cause**: Because driving forward carries risk of collision ($-25.0$) or off-road ($-20.0$), while standing still yields small lane-heading rewards without risk, the agent learned that remaining stationary maximizes survival.
* **The Goal**: Guarantee aggressive, active exploration so the policy experiences real visual speed dynamics, lane tracking, and obstacle avoidance without collapsing into idle mode.

---

## 2. Actionable TODO Checklist

### Phase 1: Reward Structure & Anti-Stalling Guards (Priority: Highest) — SUPERSEDED, implemented differently
Instead of gating each of the original shaping terms (lane centering, heading alignment, comfort,
steering smoothness, etc.) individually by speed, `src/envs/reward_calculator.py` was rewritten
around two signals: dense forward progress, and event-triggered violation penalties. Standing
still is reward-neutral (0.0), never positive, which removes the idle local optimum without
needing per-term speed gates.
- [x] **1.1 Stall Termination** — 40-step grace period, then 30-step timeout (1.5s at 20 FPS) below 2.0 km/h (exempt at red lights / near obstacles), −15.0 terminal penalty.
- [x] **1.2 No unconditional positive shaping** — the lane-centering/heading/comfort/steering terms were removed rather than speed-gated; the only positive term is progress, which is itself proportional to forward speed.
- [x] **1.3 Distance-Traveled Progress Reward** — `PROGRESS_PER_METER * forward_speed_ms * dt`, credited only while heading forward and not violating (e.g. zeroed while running a red light).

Also fixed alongside this: `DrivingStateExtractor` (`src/envs/driving_state.py`) now populates
`is_at_red_light`, `min_obs_dist`/`is_pedestrian`, `ttc_seconds`, `curve_factor`, and `is_junction`,
which were previously hardcoded to inert defaults so the obstacle/TTC/red-light terms could never
fire. The `desired_speed` unit mismatch (EasyCarla passes m/s, the reward compared it as km/h) is
also fixed, and the action space declaration now matches the actual `[-1, 1]` Tanh action mapping
used by `_sub_step` in both `camera_easycarla_env.py` and `carla_gym_env.py`.

---

### Phase 1b: Lane Discipline & Strict Termination (added after the first 10k-step eval)
The first 10k-step run on `main` produced a degenerate **full-throttle hard-left lock**
(throttle ~0.90, steer ~-0.97, sustained until a head-on crash). That policy was trained
against the hardcoded dummy state (`heading_cos = 1.0`, `lateral_dist = 0.0`), so it earned
full progress reward regardless of where or how it drove. With real state now plumbed
through, the following were added to close the remaining gaps:
- [x] **1b.1 Lane-centering penalty** — `r_lane = -0.15 * (|d_lat| / (lane_width/2))^2`,
  exempt inside junctions. Penalty-only, so it cannot resurrect the idle exploit. Scaled so
  that riding the lane edge roughly cancels the progress reward: a constant steering lock
  goes net-negative well before it reaches anything to crash into.
- [x] **1b.2 Progress reward rescaled** — `PROGRESS_PER_METER` 1.0 → 0.2 (~0.1/step at
  36 km/h), so one collision (−25) outweighs ~250 steps of ideal driving.
- [x] **1b.3 Boundary termination** — `DrivingStateExtractor.OFF_ROAD_LATERAL_LIMIT = 1.8 m`
  (or `lane_width/2 + 0.8`, whichever is stricter) and wrong-way at `heading_cos < -0.2`.
- [x] **1b.4 Zero-tolerance collision** — the 250 N·s impulse gate is gone. Any non-road
  actor (vehicle, walker, pole, building, fence, traffic light, unclassified static mesh)
  terminates on the first collision event; only gentle road/ground/terrain/sidewalk contact
  below 400 N·s is still filtered as a false positive.
- [x] **1b.5 Stall cutoff widened** — `STALL_TIMEOUT_STEPS` 30 → 80 after the 40-step grace.
- [x] **1b.6 Telemetry** — `r_lane`, `lateral_dist`, `heading_cos` added to the CSV schema
  and the trainer row; `eval_video.mp4` upload now probes the same dynamic MLflow port list
  the training supervisor uses instead of assuming 10100.

**Not yet addressed:** there is still no explicit destination or route-completion signal —
episodes end on collision, off-road, stall, or step-count truncation. "Reaching the
destination" is currently approximated by sustained violation-free forward progress.

---

### Phase 2: Action Space & Exploration Dynamics (Priority: High)
- [ ] **2.1 Increase Exploration Entropy (`--ent-coef 0.06 - 0.08`)** (`src/training/ppo_trainer.py`)
  - Raise initial entropy bonus to prevent the actor policy from collapsing to a single deterministic action.
- [ ] **2.2 Positive Throttle Mapping & Minimum Floor** (`src/envs/camera_easycarla_env.py`)
  - Prevent throttle clamping below $0.15$ during non-braking states so that every forward action produces sufficient torque to overcome CARLA's static friction.
- [ ] **2.3 Actor $\log \sigma$ Re-initialization** (`src/models/actor_critic.py`)
  - Initialize the action head with wider exploration variance ($\sigma \ge 1.0$) for continuous control.

---

### Phase 3: Curriculum & Spawn Mechanics (Priority: Medium)
- [ ] **3.1 Dynamic Spawn Cruising Speed** (`src/envs/camera_easycarla_env.py`)
  - On episode reset, spawn the ego vehicle with an initial forward speed ($10 - 15\text{ km/h}$) so it immediately experiences optical flow and steering dynamics.
- [ ] **3.2 Rolling Exploration Telemetry** (`src/logging/csv_logger.py`)
  - Log actor policy standard deviation $\sigma_{\text{throttle}}, \sigma_{\text{steer}}$ to the CSV and MLflow to monitor exploration health in real-time.
