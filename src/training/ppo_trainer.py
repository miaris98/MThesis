"""PPOTrainer: Orchestrates multi-server reinforcement learning rollouts, optimization, and checkpointing."""
import os
import time
import json
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config.training_config import TrainingConfig
from src.models.actor_critic import ActorCriticPPO
from src.envs.vector_carla_env import create_vector_carla_env
from src.logging.normalizer import RunningMeanStd
from src.logging.hardware_monitor import HardwareMonitor
from src.logging.csv_logger import CSVTelemetryLogger
from src.logging.experiment_logger import ExperimentLogger
from src.training.rollout_buffer import RolloutBuffer


class PPOTrainer:
    """High-throughput PPO Deep RL Trainer supporting parallel CARLA server environments."""

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_cuda_optimizations()

        self.ports = self.cfg.get_ports()
        self.num_envs = len(self.ports)
        self.env = create_vector_carla_env(self.cfg)
        self.agent = self._create_agent()
        try:
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.cfg.lr, foreach=False)
        except TypeError:
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.cfg.lr)
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        self.reward_normalizer = RunningMeanStd()
        self.logger = ExperimentLogger(
            self.cfg.log_dir,
            checkpoint_dir=self.cfg.checkpoint_dir,
            experiment_name=self.cfg.experiment_name,
            use_mlflow=self.cfg.use_mlflow,
            mlflow_port=self.cfg.mlflow_port,
            resume=self.cfg.resume and not self.cfg.fresh
        )
        self.logger.log_params(self.cfg)

        csv_path = os.path.join(self.cfg.log_dir, "training_telemetry.csv")
        self.csv_logger = CSVTelemetryLogger(csv_path)

        self.global_step = 0
        self.best_reward = -float("inf")
        self.episode_count = 1
        self._handle_checkpoints()

    def _init_cuda_optimizations(self) -> None:
        if torch.cuda.is_available():
            try:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    def _create_agent(self) -> ActorCriticPPO:
        agent = ActorCriticPPO(
            action_dim=3,
            features_dim=512,
            backbone_name=self.cfg.backbone,
            policy_arch=self.cfg.policy_arch,
            freeze_backbone=self.cfg.freeze_backbone,
            use_pretrained=self.cfg.use_pretrained,
            weights_path=self.cfg.weights_path
        ).to(self.device)
        return agent

    def _handle_checkpoints(self) -> None:
        if self.cfg.fresh:
            print("🧹 [START FRESH] Cleaning previous checkpoints...")
            for fname in os.listdir(self.cfg.checkpoint_dir):
                fpath = os.path.join(self.cfg.checkpoint_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        elif self.cfg.resume:
            latest_ckpt = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_latest.pth")
            if not os.path.exists(latest_ckpt):
                latest_ckpt = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_best.pth")
            if os.path.exists(latest_ckpt):
                self.agent.load_state_dict(torch.load(latest_ckpt, map_location=self.device), strict=False)
                print(f"[Resume] Loaded policy checkpoint: {latest_ckpt}")

            state_file = os.path.join(self.cfg.checkpoint_dir, "train_state.json")
            if os.path.exists(state_file):
                with open(state_file) as f:
                    st = json.load(f)
                self.global_step = st.get("global_step", 0)
                self.best_reward = st.get("best_episode_reward", -float("inf"))
                print(f"[Resume] Resuming from step {self.global_step}/{self.cfg.total_steps}")

    def _recover_env(self) -> Dict[str, np.ndarray]:
        """Seamlessly reconnect and recover parallel CARLA environments without reloading neural models."""
        print("🔄 [In-Process Recovery] Reconnecting parallel CARLA environment workers...")
        try:
            self.env.close()
        except Exception:
            pass
        time.sleep(2.0)
        for attempt in range(5):
            try:
                self.env = create_vector_carla_env(self.cfg)
                obs, _ = self.env.reset()
                print("✓ [In-Process Recovery] CARLA environment workers successfully re-established!")
                return obs
            except Exception as e:
                print(f"--> [Recovery Attempt {attempt+1}/5 Failed: {e}] Retrying in 3s...")
                time.sleep(3.0)
        raise RuntimeError("Failed to recover CARLA environment after 5 attempts.")

    def train(self) -> None:
        """Main PPO training loop with vectorized trajectory rollouts and policy updates."""
        try:
            obs, _ = self.env.reset()
        except Exception as e:
            print(f"⚠️ [Initial Environment Reset Failed: {e}] Attempting in-process recovery...")
            obs = self._recover_env()

        buffer = RolloutBuffer(
            buffer_size=self.cfg.rollout_steps,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            device=self.device,
            num_envs=self.num_envs
        )
        is_frozen = bool(getattr(self.agent.encoder, 'freeze_backbone', False))

        # Per-environment independent accumulators to prevent cross-server data mixing
        ep_rewards = np.zeros(self.num_envs, dtype=np.float32)
        ep_lengths = np.zeros(self.num_envs, dtype=int)
        ep_speeds: List[List[float]] = [[] for _ in range(self.num_envs)]
        dones = np.zeros(self.num_envs, dtype=bool)

        while self.global_step < self.cfg.total_steps:
            buffer.reset()
            rollout_start = time.time()

            warmup_steps = max(10000, int(0.20 * self.cfg.total_steps))
            curriculum_factor = min(1.0, max(0.2, self.global_step / float(warmup_steps)))
            if hasattr(self.env, 'set_curriculum_factor'):
                self.env.set_curriculum_factor(curriculum_factor)

            # 1. Rollout Collection Across Parallel CARLA Servers
            for _ in range(self.cfg.rollout_steps):
                self.global_step += self.num_envs
                img_t = torch.as_tensor(obs["image"], dtype=torch.uint8, device=self.device)
                spd_t = torch.as_tensor(obs["speed"], dtype=torch.float32, device=self.device)

                with torch.inference_mode():
                    vis_feat = self.agent.extract_visual_features(img_t) if is_frozen else None
                    action, log_prob, _, value = self.agent.get_action_and_value(
                        image=img_t if not is_frozen else None,
                        speed=spd_t,
                        visual_features=vis_feat
                    )

                action_np = action.cpu().numpy()
                try:
                    next_obs, rewards, term, trunc, infos = self.env.step(action_np)
                except Exception as e:
                    print(f"⚠️ [Environment Step Exception: {e}] Triggering in-process environment recovery...")
                    obs = self._recover_env()
                    continue

                dones = term | trunc

                raw_r = np.array(rewards, dtype=np.float32)
                clipped_r = np.clip(raw_r, -self.cfg.reward_clip, self.cfg.reward_clip)
                self.reward_normalizer.update(clipped_r)
                norm_r = clipped_r / self.reward_normalizer.std

                buffer.add(
                    speed=spd_t, action=action, log_prob=log_prob,
                    reward=norm_r, done=dones, value=value,
                    obs_img=img_t if not is_frozen else None,
                    obs_vis=vis_feat if is_frozen else None
                )

                hw = HardwareMonitor.get_metrics()

                # Process telemetry and lifecycle independently per environment worker
                for e in range(self.num_envs):
                    ep_rewards[e] += float(raw_r[e])
                    ep_lengths[e] += 1
                    info_e = infos[e]
                    spd_val = info_e.get("speed_kmh", float(next_obs["speed"][e][0]))
                    ep_speeds[e].append(spd_val)

                    self.csv_logger.log_step({
                        "global_step": self.global_step,
                        "env_id": e,
                        "episode": self.episode_count,
                        "step_in_ep": ep_lengths[e],
                        "speed_kmh": round(float(spd_val), 2),
                        "action_throttle": round(float(action_np[e, 0]), 3),
                        "action_steer": round(float(action_np[e, 1]), 3),
                        "action_brake": round(float(action_np[e, 2]), 3),
                        "raw_reward": round(float(raw_r[e]), 4),
                        "normalized_reward": round(float(norm_r[e]), 4),
                        "curriculum_alpha": round(float(curriculum_factor), 2),
                        **hw,
                        "is_collision": info_e.get("is_collision", False),
                        "is_off_road": info_e.get("is_off_road", False),
                        "termination_reason": info_e.get("termination_reason", "") if dones[e] else ""
                    })

                    if dones[e]:
                        self._on_episode_done(e, float(ep_rewards[e]), ep_speeds[e], info_e)
                        ep_rewards[e] = 0.0
                        ep_lengths[e] = 0
                        ep_speeds[e] = []

                obs = next_obs

            # 2. Advantage Estimation & PPO Update
            self._update_ppo(buffer, obs, dones, is_frozen, rollout_start)

        self._shutdown()

    def _on_episode_done(self, env_id: int, ep_reward: float, ep_speeds: list, info: dict) -> None:
        self.episode_count += 1
        self.csv_logger.flush()
        avg_speed = np.mean(ep_speeds) if ep_speeds else 0.0
        reason = info.get("termination_reason", "Finished")
        print(f"[Step {self.global_step:05d}/{self.cfg.total_steps} | Env #{env_id}] Episode Finished | Reward: {ep_reward:+.2f} | Avg Speed: {avg_speed:.1f} km/h | Reason: {reason}")
        self.logger.add_scalar("Reward/Episode_Total", ep_reward, self.global_step)
        self.logger.add_scalar("Speed/Avg_kmh", avg_speed, self.global_step)
        self.logger.add_scalar(f"Reward/Env_{env_id}_Total", ep_reward, self.global_step)

        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
            best_path = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_best.pth")
            torch.save(self.agent.state_dict(), best_path)

    def _update_ppo(self, buffer: RolloutBuffer, last_obs: dict, last_dones: np.ndarray, is_frozen: bool, rollout_start: float) -> None:
        with torch.inference_mode():
            last_img = torch.as_tensor(last_obs["image"], dtype=torch.uint8, device=self.device)
            last_spd = torch.as_tensor(last_obs["speed"], dtype=torch.float32, device=self.device)
            last_vis = self.agent.extract_visual_features(last_img) if is_frozen else None
            next_val = self.agent.get_action_and_value(speed=last_spd, visual_features=last_vis)[3]

        b_vis, b_spd, b_act, b_logp, b_adv, b_ret, b_val = buffer.compute_returns_and_advantages(next_val, next_done=last_dones)

        total_samples = buffer.total_transitions
        b_inds = np.arange(total_samples)
        policy_losses, value_losses = [], []
        ppo_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, total_samples, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb = b_inds[start:end]

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    _, new_logp, entropy, new_val = self.agent.get_action_and_value(
                        speed=b_spd[mb], action=b_act[mb], visual_features=b_vis[mb] if is_frozen else None
                    )
                    ratio = (new_logp - b_logp[mb]).exp()
                    pg_loss = torch.max(-b_adv[mb] * ratio, -b_adv[mb] * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)).mean()
                    v_loss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()
                    total_loss = pg_loss + 0.5 * v_loss - self.cfg.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                policy_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sps = total_samples / max(1e-4, time.time() - rollout_start)
        ppo_elapsed = time.time() - ppo_start
        print(f"--- Rollout Update Complete | Step: {self.global_step}/{self.cfg.total_steps} | SPS: {sps:.1f} | PPO Opt: {ppo_elapsed*1000.0:.1f}ms | Loss: {np.mean(policy_losses):.4f} ---")

        latest_path = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_latest.pth")
        torch.save(self.agent.state_dict(), latest_path)
        with open(os.path.join(self.cfg.checkpoint_dir, "train_state.json"), "w") as f:
            json.dump({"global_step": self.global_step, "best_episode_reward": float(self.best_reward)}, f, indent=2)

    def _shutdown(self) -> None:
        self.env.close()
        self.csv_logger.close()
        self.logger.close()
        print("✓ Training Completed Successfully!")
