# CARLA PPO Driving Policy & Exploration Improvement Roadmap

## 1. Executive Summary & Problem Diagnosis
During RL training, the agent fell into a passive local optimum: **holding throttle at ~1% while steering slightly right**.
* **The Root Cause**: Because driving forward carries risk of collision ($-25.0$) or off-road ($-20.0$), while standing still yields small lane-heading rewards without risk, the agent learned that remaining stationary maximizes survival.
* **The Goal**: Guarantee aggressive, active exploration so the policy experiences real visual speed dynamics, lane tracking, and obstacle avoidance without collapsing into idle mode.

---

## 2. Actionable TODO Checklist

### Phase 1: Reward Structure & Anti-Stalling Guards (Priority: Highest)
- [ ] **1.1 Strict 30-Step Stall Termination** (`src/envs/reward_calculator.py`)
  - Reduce stall timeout from 120 steps down to **30 steps (1.5 seconds at 20 FPS)** whenever speed $< 2.0\text{ km/h}$ (unless at a red light or obstacle).
  - Standing still becomes an immediate failure with a $-15.0$ penalty.
- [ ] **1.2 Speed-Gated Positive Rewards** (`src/envs/reward_calculator.py`)
  - Gate **ALL** positive rewards (lane centering, heading alignment) by forward speed:
    $$R_{\text{step}} = R_{\text{base}} \times \min\left(1.0, \frac{v_{\text{proj}}}{v_{\text{target}}}\right)$$
  - Standing still ($v = 0$) will yield **$0.0$ positive reward**.
- [ ] **1.3 Distance-Traveled Longitudinal Bonus** (`src/envs/reward_calculator.py`)
  - Add explicit progress reward: $+0.1$ per meter traveled along the lane centerline.

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
