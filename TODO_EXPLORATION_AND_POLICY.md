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
