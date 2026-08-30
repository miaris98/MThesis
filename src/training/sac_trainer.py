"""SACTrainer: off-policy Soft Actor-Critic training over cached visual features in CARLA."""
import os
import time
import json
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config.training_config import TrainingConfig
from src.models.sac_networks import SACActorCritic
from src.envs.vector_carla_env import create_vector_carla_env
from src.logging.hardware_monitor import HardwareMonitor
from src.logging.csv_logger import CSVTelemetryLogger
from src.logging.experiment_logger import ExperimentLogger
from src.training.replay_buffer import ReplayBuffer
from src.training.trainer_telemetry import TelemetryMixin


class SACTrainer(TelemetryMixin):
    """
    Soft Actor-Critic trainer for continuous CARLA control.

    Unlike the on-policy PPO trainer, every collected transition is written to a replay
    buffer and resampled many times, which is the point: CARLA simulation - not gradient
    computation - is the throughput bottleneck on this setup.
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        if not self.cfg.freeze_backbone:
            raise ValueError(
                "SAC requires --freeze-backbone. Replaying raw frames for a trainable backbone "
                "would need tens of GB of buffer; the cached-feature buffer assumes a frozen encoder."
            )

        self.ports = self.cfg.get_ports()
        self.num_envs = len(self.ports)
        self.env = create_vector_carla_env(self.cfg)

        self.agent = SACActorCritic(
            action_dim=3, features_dim=512, backbone_name=self.cfg.backbone,
            freeze_backbone=True, use_pretrained=self.cfg.use_pretrained,
            weights_path=self.cfg.weights_path
        ).to(self.device)

        # Explicit: pins the frozen backbone's BatchNorm layers to eval so cached replay
        # features stay consistent for the whole run.
        self.agent.train()

        self.actor_opt = optim.Adam(self.agent.actor_parameters(), lr=self.cfg.lr)
        self.critic_opt = optim.Adam(self.agent.critic_parameters(), lr=self.cfg.lr)

        # Automatic entropy temperature: target entropy of -|A| is the SAC default heuristic.
        self.target_entropy = -float(self.agent.action_dim)
        self.log_alpha = torch.tensor(
            float(np.log(self.cfg.init_alpha)), device=self.device, requires_grad=self.cfg.autotune_alpha
        )
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.cfg.lr) if self.cfg.autotune_alpha else None

        self.buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size, visual_dim=self.agent.visual_dim,
            action_dim=3, device=self.device
        )
        print(f"Replay buffer: {self.cfg.buffer_size:,} transitions ~ {self.buffer.memory_mb:.0f} MB (CPU)")

        self.logger = ExperimentLogger(
            self.cfg.log_dir, checkpoint_dir=self.cfg.checkpoint_dir,
            experiment_name=self.cfg.experiment_name, use_mlflow=self.cfg.use_mlflow,
            mlflow_port=self.cfg.mlflow_port, resume=self.cfg.resume and not self.cfg.fresh
        )
        self.logger.log_params(self.cfg)
        self.csv_logger = CSVTelemetryLogger(os.path.join(self.cfg.log_dir, "training_telemetry.csv"))

        self.global_step, self.episode_count = 0, 1
        self.best_reward, self.best_moving_avg, self.patience_counter = -float("inf"), -float("inf"), 0
        self.recent_rewards: List[float] = []
        self.early_stop_triggered, self.early_stop_reason = False, ""
        self.train_start_time, self.last_progress_step, self.last_progress_time = None, 0, None
        self.last_q_loss, self.last_pi_loss, self.last_alpha_loss, self.last_entropy = None, None, None, None
        self.last_sps, self.last_fps = 0.0, 0.0
        self._handle_checkpoints()

    def _handle_checkpoints(self) -> None:
        """Clear checkpoints on --fresh, or restore actor/critic weights and step count on --resume."""
        if self.cfg.fresh:
            for name in os.listdir(self.cfg.checkpoint_dir):
                path = os.path.join(self.cfg.checkpoint_dir, name)
                if os.path.isfile(path):
                    try: os.remove(path)
                    except Exception: pass
            return
        if not self.cfg.resume:
            return
        latest = os.path.join(self.cfg.checkpoint_dir, "sac_carla_latest.pth")
        if not os.path.exists(latest):
            latest = os.path.join(self.cfg.checkpoint_dir, "sac_carla_best.pth")
        if os.path.exists(latest):
            self.agent.load_state_dict(torch.load(latest, map_location=self.device), strict=False)
            print(f"Restored SAC weights from {latest}")
        state_file = os.path.join(self.cfg.checkpoint_dir, "sac_train_state.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                st = json.load(f)
            self.global_step = st.get("global_step", 0)
            self.best_reward = st.get("best_episode_reward", -float("inf"))

    def _recover_env(self) -> Dict[str, np.ndarray]:
        """Tear down and rebuild the vectorized CARLA env after an unrecoverable step error."""
        try: self.env.close()
        except Exception: pass
        time.sleep(2.0)
        for attempt in range(5):
            try:
                self.env = create_vector_carla_env(self.cfg)
                obs, _ = self.env.reset()
                return obs
            except Exception as e:
                print(f"--> [Recovery Attempt {attempt+1}/5: {e}] Retrying in 3s...")
                time.sleep(3.0)
        raise RuntimeError("Failed to recover CARLA environment.")

    @property
    def alpha(self) -> float:
        """Current entropy temperature."""
        return float(self.log_alpha.exp().item())

    def train(self) -> None:
        """Main SAC loop: collect one step per env, then run gradient updates against the replay."""
        try:
            obs, _ = self.env.reset()
        except Exception as e:
            print(f"[Reset Failed: {e}] Recovering CARLA env...")
            obs = self._recover_env()

        self.train_start_time = time.time()
        self.last_progress_time = time.time()
        self.last_progress_step = self.global_step

        ep_rewards = np.zeros(self.num_envs, dtype=np.float32)
        ep_lengths = np.zeros(self.num_envs, dtype=int)
        ep_speeds: List[List[float]] = [[] for _ in range(self.num_envs)]

        while self.global_step < self.cfg.total_steps:
            warmup_steps = max(10000, int(0.20 * self.cfg.total_steps))
            curriculum_factor = min(1.0, max(0.2, self.global_step / float(warmup_steps)))
            if hasattr(self.env, 'set_curriculum_factor'):
                self.env.set_curriculum_factor(curriculum_factor)

            img_t = torch.as_tensor(obs["image"], dtype=torch.uint8, device=self.device)
            spd_t = torch.as_tensor(obs["speed"], dtype=torch.float32, device=self.device)

            with torch.inference_mode():
                vis_feat = self.agent.extract_visual_features(img_t)
                if self.global_step < self.cfg.learning_starts:
                    # Uniform random warm-start fills the buffer with genuinely diverse behavior
                    # before the policy has any signal worth following.
                    action = torch.rand((self.num_envs, 3), device=self.device) * 2.0 - 1.0
                else:
                    action, _ = self.agent.sample_action(vis_feat, spd_t)

            action_np = action.cpu().numpy()
            try:
                next_obs, rewards, term, trunc, infos = self.env.step(action_np)
            except Exception as e:
                print(f"[Step Exception: {e}] Triggering env recovery...")
                obs = self._recover_env()
                continue

            self.global_step += self.num_envs
            dones = term | trunc
            raw_r = np.array(rewards, dtype=np.float32)
            stored_r = np.clip(raw_r, -self.cfg.reward_clip, self.cfg.reward_clip)

            next_img_t = torch.as_tensor(next_obs["image"], dtype=torch.uint8, device=self.device)
            next_spd_t = torch.as_tensor(next_obs["speed"], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                next_vis_feat = self.agent.extract_visual_features(next_img_t)

            # Truncation is a time limit, not a real terminal state: bootstrapping must continue
            # through it, so only genuine terminations mask the next-state value.
            self.buffer.add(
                visual=vis_feat, speed=spd_t, action=action, reward=stored_r,
                next_visual=next_vis_feat, next_speed=next_spd_t,
                done=np.asarray(term, dtype=np.float32)
            )

            if self.global_step >= self.cfg.learning_starts:
                for _ in range(self.cfg.updates_per_step * self.num_envs):
                    self._update_sac()

            hw = HardwareMonitor.get_metrics()
            for e in range(self.num_envs):
                ep_rewards[e] += float(raw_r[e])
                ep_lengths[e] += 1
                info_e = infos[e]
                spd_val = info_e.get("speed_kmh", float(next_obs["speed"][e][0]))
                ep_speeds[e].append(spd_val)
                self.log_telemetry_row(
                    env_id=e, info=info_e, action_np=action_np, raw_reward=raw_r[e],
                    stored_reward=stored_r[e], curriculum_factor=curriculum_factor,
                    step_in_ep=ep_lengths[e], speed_kmh=spd_val, done=bool(dones[e]),
                    hardware=hw, extra=self._sac_metrics())

                if dones[e]:
                    self._on_episode_done(e, float(ep_rewards[e]), ep_speeds[e], info_e)
                    ep_rewards[e], ep_lengths[e], ep_speeds[e] = 0.0, 0, []

            self.report_progress(suffix=f" | Buffer: {len(self.buffer):,} | alpha: {self.alpha:.3f}")
            obs = next_obs

            if self.early_stop_triggered:
                print(f"\n[{time.strftime('%H:%M:%S')} | EARLY STOPPING] {self.early_stop_reason}", flush=True)
                break

        self._shutdown()

    def _update_sac(self) -> None:
        """One SAC gradient step: twin-critic Bellman backup, policy improvement, temperature tuning."""
        vis, spd, act, rew, next_vis, next_spd, done = self.buffer.sample(self.cfg.sac_batch_size)
        alpha = self.log_alpha.exp().detach()

        # --- Critics: entropy-regularized Bellman target from the twin target networks ---
        with torch.no_grad():
            next_action, next_logp = self.agent.sample_action(next_vis, next_spd)
            q1_t, q2_t = self.agent.target_q_values(next_vis, next_spd, next_action)
            min_q_next = torch.min(q1_t, q2_t) - alpha * next_logp
            q_target = rew + self.cfg.gamma * (1.0 - done) * min_q_next

        q1, q2 = self.agent.q_values(vis, spd, act)
        critic_loss = nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.critic_parameters(), 1.0)
        self.critic_opt.step()

        # --- Actor: maximize Q while staying as stochastic as the temperature allows ---
        new_action, logp = self.agent.sample_action(vis, spd)
        q1_pi, q2_pi = self.agent.q_values(vis, spd, new_action)
        actor_loss = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.agent.actor_parameters(), 1.0)
        self.actor_opt.step()

        # --- Temperature: drive average policy entropy toward target_entropy ---
        alpha_loss_val = 0.0
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha.exp() * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_val = float(alpha_loss.item())

        self.agent.soft_update_targets(self.cfg.tau)
        self.last_q_loss = float(critic_loss.item())
        self.last_pi_loss = float(actor_loss.item())
        self.last_alpha_loss = alpha_loss_val
        self.last_entropy = float(-logp.mean().item())

    def _sac_metrics(self) -> dict:
        """SAC-specific loss columns appended to the shared telemetry schema."""
        return {
            "loss_policy": round(self.last_pi_loss, 4) if self.last_pi_loss is not None else "",
            "loss_value": round(self.last_q_loss, 4) if self.last_q_loss is not None else "",
            "loss_entropy": round(self.last_entropy, 4) if self.last_entropy is not None else "",
            "sac_alpha": round(self.alpha, 4),
            "sac_alpha_loss": round(self.last_alpha_loss, 4) if self.last_alpha_loss is not None else ""
        }

    def _on_episode_done(self, env_id: int, ep_reward: float, ep_speeds: list, info: dict) -> None:
        """Record episode statistics, checkpoint on improvement, and evaluate early stopping."""
        self.episode_count += 1
        self.recent_rewards.append(ep_reward)
        if len(self.recent_rewards) > self.cfg.early_stopping_window * 2:
            self.recent_rewards.pop(0)
        ma_reward = float(np.mean(self.recent_rewards[-self.cfg.early_stopping_window:]))

        self.csv_logger.flush()
        avg_speed = float(np.mean(ep_speeds)) if ep_speeds else 0.0
        reason = info.get("termination_reason", "Finished")
        print(f"[{time.strftime('%H:%M:%S')} | Step {self.global_step:05d}/{self.cfg.total_steps} | "
              f"Env #{env_id}] Episode #{self.episode_count:03d} | R: {ep_reward:+.2f} "
              f"(MA-{self.cfg.early_stopping_window}: {ma_reward:+.2f}) | "
              f"Spd: {avg_speed:4.1f} km/h | {reason}", flush=True)

        self.logger.add_scalar("Reward/Episode_Total", ep_reward, self.global_step)
        self.logger.add_scalar("Reward/Moving_Average", ma_reward, self.global_step)
        self.logger.add_scalar("Speed/Avg_kmh", avg_speed, self.global_step)
        self.logger.add_scalar("SAC/Alpha", self.alpha, self.global_step)
        if self.last_entropy is not None:
            self.logger.add_scalar("SAC/Policy_Entropy", self.last_entropy, self.global_step)
        if self.last_q_loss is not None:
            self.logger.add_scalar("SAC/Critic_Loss", self.last_q_loss, self.global_step)

        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
            torch.save(self.agent.state_dict(), os.path.join(self.cfg.checkpoint_dir, "sac_carla_best.pth"))

        if self.cfg.early_stopping and len(self.recent_rewards) >= max(2, self.cfg.early_stopping_window // 2):
            if ma_reward > self.best_moving_avg + self.cfg.early_stopping_min_delta:
                self.best_moving_avg, self.patience_counter = ma_reward, 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.early_stopping_patience:
                    self.early_stop_triggered = True
                    self.early_stop_reason = (f"Plateau: {self.patience_counter} episodes without "
                                              f"improvement (Best MA: {self.best_moving_avg:.2f}, "
                                              f"Cur: {ma_reward:.2f})")
        if self.cfg.target_reward is not None and ma_reward >= self.cfg.target_reward:
            self.early_stop_triggered = True
            self.early_stop_reason = f"Target reward achieved: MA {ma_reward:.2f} >= {self.cfg.target_reward:.2f}"

    def _shutdown(self) -> None:
        """Persist final weights and training state, then release the environment and loggers."""
        torch.save(self.agent.state_dict(), os.path.join(self.cfg.checkpoint_dir, "sac_carla_latest.pth"))
        with open(os.path.join(self.cfg.checkpoint_dir, "sac_train_state.json"), "w") as f:
            json.dump({"global_step": self.global_step, "best_episode_reward": self.best_reward,
                       "episode_count": self.episode_count}, f, indent=2)
        self.csv_logger.close()
        try: self.logger.close()
        except Exception: pass
        try: self.env.close()
        except Exception: pass
        print(f"SAC training finished at step {self.global_step:,} | "
              f"Best episode reward: {self.best_reward:+.2f}")
