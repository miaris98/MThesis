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
        self._init_cuda()
        self.ports = self.cfg.get_ports()
        self.num_envs = len(self.ports)
        self.env = create_vector_carla_env(self.cfg)
        self.agent = ActorCriticPPO(
            action_dim=3, features_dim=512, backbone_name=self.cfg.backbone,
            policy_arch=self.cfg.policy_arch, freeze_backbone=self.cfg.freeze_backbone,
            use_pretrained=self.cfg.use_pretrained, weights_path=self.cfg.weights_path
        ).to(self.device)
        try:
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.cfg.lr, foreach=False)
        except TypeError:
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.cfg.lr)
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.reward_normalizer = RunningMeanStd()
        self.logger = ExperimentLogger(
            self.cfg.log_dir, checkpoint_dir=self.cfg.checkpoint_dir,
            experiment_name=self.cfg.experiment_name, use_mlflow=self.cfg.use_mlflow,
            mlflow_port=self.cfg.mlflow_port, resume=self.cfg.resume and not self.cfg.fresh
        )
        self.logger.log_params(self.cfg)
        self.csv_logger = CSVTelemetryLogger(os.path.join(self.cfg.log_dir, "training_telemetry.csv"))
        self.global_step, self.best_reward, self.best_moving_avg, self.patience_counter = 0, -float("inf"), -float("inf"), 0
        self.recent_rewards, self.early_stop_triggered, self.early_stop_reason = [], False, ""
        self.episode_count, self.train_start_time, self.last_progress_step, self.last_progress_time = 1, None, 0, None
        self.last_p_loss, self.last_v_loss, self.last_entropy = None, None, None
        self.last_kl, self.last_clip_frac, self.last_expl_var = None, None, None
        self.last_sps, self.last_fps = 0.0, 0.0
        self.last_artifact_sync_step = 0
        self.artifact_sync_interval = 5000
        self._handle_checkpoints()

    def _init_cuda(self) -> None:
        if torch.cuda.is_available():
            try:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    def _handle_checkpoints(self) -> None:
        if self.cfg.fresh:
            for f in [os.path.join(self.cfg.checkpoint_dir, x) for x in os.listdir(self.cfg.checkpoint_dir)]:
                if os.path.isfile(f):
                    try: os.remove(f)
                    except Exception: pass
        elif self.cfg.resume:
            latest = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_latest.pth")
            if not os.path.exists(latest):
                latest = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_best.pth")
            if os.path.exists(latest):
                self.agent.load_state_dict(torch.load(latest, map_location=self.device), strict=False)
            sf = os.path.join(self.cfg.checkpoint_dir, "train_state.json")
            if os.path.exists(sf):
                with open(sf) as f: st = json.load(f)
                self.global_step, self.best_reward = st.get("global_step", 0), st.get("best_episode_reward", -float("inf"))

    def _recover_env(self) -> Dict[str, np.ndarray]:
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

    def train(self) -> None:
        """Main PPO training loop with vectorized rollouts, advantage estimation, and early stopping."""
        try:
            obs, _ = self.env.reset()
        except Exception as e:
            print(f"⚠️ [Reset Failed: {e}] Recovering CARLA env...")
            obs = self._recover_env()

        self.train_start_time = time.time()
        self.last_progress_time = time.time()
        self.last_progress_step = self.global_step

        buffer = RolloutBuffer(
            buffer_size=self.cfg.rollout_steps, gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda, device=self.device, num_envs=self.num_envs
        )
        is_frozen = bool(getattr(self.agent.encoder, 'freeze_backbone', False))
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

            for _ in range(self.cfg.rollout_steps):
                self.global_step += self.num_envs
                img_t = torch.as_tensor(obs["image"], dtype=torch.uint8, device=self.device)
                spd_t = torch.as_tensor(obs["speed"], dtype=torch.float32, device=self.device)

                with torch.inference_mode():
                    vis_feat = self.agent.extract_visual_features(img_t) if is_frozen else None
                    action, log_prob, _, value = self.agent.get_action_and_value(
                        image=img_t if not is_frozen else None, speed=spd_t, visual_features=vis_feat
                    )

                action_np = action.cpu().numpy()
                try:
                    next_obs, rewards, term, trunc, infos = self.env.step(action_np)
                except Exception as e:
                    print(f"⚠️ [Step Exception: {e}] Triggering env recovery...")
                    obs = self._recover_env()
                    continue

                dones = term | trunc
                raw_r = np.array(rewards, dtype=np.float32)
                clipped_r = np.clip(raw_r, -self.cfg.reward_clip, self.cfg.reward_clip)
                self.reward_normalizer.update(clipped_r)
                norm_r = clipped_r / self.reward_normalizer.std

                buffer.add(
                    speed=spd_t, action=action, log_prob=log_prob, reward=norm_r, done=dones,
                    value=value, obs_img=img_t if not is_frozen else None, obs_vis=vis_feat if is_frozen else None
                )

                hw = HardwareMonitor.get_metrics()
                for e in range(self.num_envs):
                    ep_rewards[e] += float(raw_r[e])
                    ep_lengths[e] += 1
                    info_e = infos[e]
                    spd_val = info_e.get("speed_kmh", float(next_obs["speed"][e][0]))
                    ep_speeds[e].append(spd_val)

                    self.csv_logger.log_step({
                        "global_step": self.global_step, "env_id": e, "episode": self.episode_count,
                        "step_in_ep": ep_lengths[e], "speed_kmh": round(float(spd_val), 2),
                        "action_throttle": round(float(action_np[e, 0]), 3),
                        "action_steer": round(float(action_np[e, 1]), 3),
                        "action_brake": round(float(action_np[e, 2]), 3),
                        "raw_reward": round(float(raw_r[e]), 4),
                        "normalized_reward": round(float(norm_r[e]), 4),
                        "curriculum_alpha": round(float(curriculum_factor), 2),
                        "r_progress": round(float(info_e.get("r_progress", 0.0)), 3),
                        "r_light": round(float(info_e.get("r_light", 0.0)), 3),
                        "r_obstacle": round(float(info_e.get("r_obstacle", 0.0)), 3),
                        "r_ttc": round(float(info_e.get("r_ttc", 0.0)), 3),
                        "r_terminal": round(float(info_e.get("r_terminal", 0.0)), 3),
                        "loss_policy": round(self.last_p_loss, 4) if self.last_p_loss is not None else "",
                        "loss_value": round(self.last_v_loss, 4) if self.last_v_loss is not None else "",
                        "loss_entropy": round(self.last_entropy, 4) if self.last_entropy is not None else "",
                        "loss_approx_kl": round(self.last_kl, 4) if self.last_kl is not None else "",
                        "loss_clip_fraction": round(self.last_clip_frac, 4) if self.last_clip_frac is not None else "",
                        "loss_explained_variance": round(self.last_expl_var, 4) if self.last_expl_var is not None else "",
                        "sps": round(self.last_sps, 1) if self.last_sps else "",
                        "fps": round(self.last_fps, 1) if self.last_fps else "",
                        **hw, "is_collision": info_e.get("is_collision", False),
                        "is_off_road": info_e.get("is_off_road", False),
                        "termination_reason": info_e.get("termination_reason", "") if dones[e] else ""
                    })

                    if dones[e]:
                        self._on_episode_done(e, float(ep_rewards[e]), ep_speeds[e], info_e)
                        ep_rewards[e] = 0.0
                        ep_lengths[e] = 0
                        ep_speeds[e] = []

                if (self.global_step - self.last_progress_step) >= max(20, self.num_envs * 10):
                    now_str = time.strftime("%H:%M:%S")
                    s_delta = self.global_step - self.last_progress_step
                    t_delta = max(1e-5, time.time() - (self.last_progress_time or time.time()))
                    sps = s_delta / t_delta
                    fps = sps * self.cfg.frame_skip
                    pct = min(100.0, 100.0 * self.global_step / float(self.cfg.total_steps))
                    print(f"[{now_str} | Step {self.global_step:05d}/{self.cfg.total_steps} ({pct:4.1f}%) | {sps:4.1f} Steps/s ({fps:4.1f} FPS) | Episodes: {self.episode_count}]", flush=True)
                    self.last_progress_step = self.global_step
                    self.last_progress_time = time.time()

                obs = next_obs

            self._update_ppo(buffer, obs, dones, is_frozen, rollout_start)

            if self.cfg.early_stopping and len(self.recent_rewards) >= max(2, self.cfg.early_stopping_window // 2):
                cur_ma = float(np.mean(self.recent_rewards[-self.cfg.early_stopping_window:]))
                if cur_ma > self.best_moving_avg + self.cfg.early_stopping_min_delta:
                    self.best_moving_avg = cur_ma
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.cfg.early_stopping_patience:
                        self.early_stop_triggered = True
                        self.early_stop_reason = f"Plateau: {self.patience_counter} rollouts without reward improvement (Best MA: {self.best_moving_avg:.2f}, Cur: {cur_ma:.2f})"

            if self.early_stop_triggered:
                print(f"\n🛑 [{time.strftime('%H:%M:%S')} | EARLY STOPPING] {self.early_stop_reason}", flush=True)
                break

        self._shutdown()

    def _on_episode_done(self, env_id: int, ep_reward: float, ep_speeds: list, info: dict) -> None:
        self.episode_count += 1
        self.recent_rewards.append(ep_reward)
        if len(self.recent_rewards) > self.cfg.early_stopping_window * 2:
            self.recent_rewards.pop(0)
        ma_reward = float(np.mean(self.recent_rewards[-self.cfg.early_stopping_window:]))

        self.csv_logger.flush()
        avg_speed = np.mean(ep_speeds) if ep_speeds else 0.0
        reason = info.get("termination_reason", "Finished")
        now_str = time.strftime("%H:%M:%S")
        elapsed = max(1e-5, time.time() - (self.train_start_time or time.time()))
        overall_sps = self.global_step / elapsed
        overall_fps = overall_sps * self.cfg.frame_skip
        print(f"[{now_str} | Step {self.global_step:05d}/{self.cfg.total_steps} | Env #{env_id}] Episode #{self.episode_count:03d} | R: {ep_reward:+.2f} (MA-{self.cfg.early_stopping_window}: {ma_reward:+.2f}) | Spd: {avg_speed:4.1f} km/h | {reason}", flush=True)
        self.logger.add_scalar("Reward/Episode_Total", ep_reward, self.global_step)
        self.logger.add_scalar("Reward/Moving_Average", ma_reward, self.global_step)
        self.logger.add_scalar("Speed/Avg_kmh", avg_speed, self.global_step)
        self.logger.add_scalar(f"Reward/Env_{env_id}_Total", ep_reward, self.global_step)
        self.logger.add_scalar("Perf/Steps_Per_Second", overall_sps, self.global_step)
        self.logger.add_scalar("Perf/Frames_Per_Second", overall_fps, self.global_step)

        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
            best_path = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_best.pth")
            torch.save(self.agent.state_dict(), best_path)

        if self.cfg.target_reward is not None and ma_reward >= self.cfg.target_reward:
            self.early_stop_triggered = True
            self.early_stop_reason = f"Target reward achieved: MA {ma_reward:.2f} >= {self.cfg.target_reward:.2f}"

    def _update_ppo(self, buffer: RolloutBuffer, last_obs: dict, last_dones: np.ndarray, is_frozen: bool, rollout_start: float) -> None:
        with torch.inference_mode():
            last_img = torch.as_tensor(last_obs["image"], dtype=torch.uint8, device=self.device)
            last_spd = torch.as_tensor(last_obs["speed"], dtype=torch.float32, device=self.device)
            last_vis = self.agent.extract_visual_features(last_img) if is_frozen else None
            next_val = self.agent.get_action_and_value(speed=last_spd, visual_features=last_vis)[3]

        b_vis, b_spd, b_act, b_logp, b_adv, b_ret, b_val = buffer.compute_returns_and_advantages(next_val, next_done=last_dones)
        total_samples = buffer.total_transitions
        b_inds = np.arange(total_samples)
        policy_losses, value_losses, entropies, kls, clip_fracs = [], [], [], [], []
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
                    log_ratio = new_logp - b_logp[mb]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean().item()

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
                entropies.append(entropy.mean().item())
                kls.append(approx_kl)
                clip_fracs.append(clip_frac)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        now_str = time.strftime("%H:%M:%S")
        sps = total_samples / max(1e-4, time.time() - rollout_start)
        fps = sps * self.cfg.frame_skip
        ppo_elapsed = time.time() - ppo_start
        mean_p_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
        mean_v_loss = float(np.mean(value_losses)) if value_losses else 0.0
        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
        mean_kl = float(np.mean(kls)) if kls else 0.0
        mean_clip_frac = float(np.mean(clip_fracs)) if clip_fracs else 0.0

        with torch.no_grad():
            y_true = b_ret.cpu().numpy()
            y_pred = b_val.cpu().numpy()
            var_y = np.var(y_true)
            expl_var = float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 0 else 0.0

        self.logger.add_scalar("Loss/Policy_Loss", mean_p_loss, self.global_step)
        self.logger.add_scalar("Loss/Value_Loss", mean_v_loss, self.global_step)
        self.logger.add_scalar("Loss/Entropy", mean_entropy, self.global_step)
        self.logger.add_scalar("Loss/Approx_KL", mean_kl, self.global_step)
        self.logger.add_scalar("Loss/Clip_Fraction", mean_clip_frac, self.global_step)
        self.logger.add_scalar("Loss/Explained_Variance", expl_var, self.global_step)
        self.logger.add_scalar("Perf/PPO_Optimization_MS", ppo_elapsed * 1000.0, self.global_step)

        print(f"[{now_str} | PPO Policy Update] Step: {self.global_step:05d}/{self.cfg.total_steps} | Rollout: {sps:4.1f} Steps/s ({fps:4.1f} FPS) | PPO Opt: {ppo_elapsed*1000.0:5.1f}ms | Policy Loss: {mean_p_loss:+.4f} | Value Loss: {mean_v_loss:.4f} | KL: {mean_kl:.4f} | ExplVar: {expl_var:.2f}", flush=True)

        self.last_p_loss = mean_p_loss
        self.last_v_loss = mean_v_loss
        self.last_entropy = mean_entropy
        self.last_kl = mean_kl
        self.last_clip_frac = mean_clip_frac
        self.last_expl_var = expl_var
        self.last_sps = sps
        self.last_fps = fps

        latest_path = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_latest.pth")
        torch.save(self.agent.state_dict(), latest_path)
        with open(os.path.join(self.cfg.checkpoint_dir, "train_state.json"), "w") as f:
            json.dump({"global_step": self.global_step, "best_episode_reward": float(self.best_reward)}, f, indent=2)

        # Periodic MLflow artifact update (every 5000 steps)
        self._sync_artifacts(force=False)

    def _sync_artifacts(self, force: bool = False) -> None:
        """Sync telemetry CSV and checkpoints to MLflow artifacts every 5000 steps or at shutdown."""
        if not force and (self.global_step - self.last_artifact_sync_step) < self.artifact_sync_interval:
            return

        self.last_artifact_sync_step = self.global_step
        if hasattr(self, 'csv_logger'):
            self.csv_logger.flush()
            if os.path.exists(self.csv_logger.filepath):
                self.logger.log_artifact(self.csv_logger.filepath)

        best_path = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_best.pth")
        if os.path.exists(best_path):
            self.logger.log_artifact(best_path)
        latest_path = os.path.join(self.cfg.checkpoint_dir, "ppo_carla_latest.pth")
        if os.path.exists(latest_path):
            self.logger.log_artifact(latest_path)
        state_path = os.path.join(self.cfg.checkpoint_dir, "train_state.json")
        if os.path.exists(state_path):
            self.logger.log_artifact(state_path)

        # Sync evaluation videos if present
        for v_path in ["/workspace/eval_video.mp4", "./eval_video.mp4", "/workspace/eval_video_best.mp4"]:
            if os.path.exists(v_path):
                self.logger.log_artifact(v_path)

        now_str = time.strftime("%H:%M:%S")
        print(f"[{now_str}] 📦 [MLflow Artifact Sync] Synced training_telemetry.csv, model checkpoints & artifacts at step {self.global_step:05d}", flush=True)

    def _shutdown(self) -> None:
        self.env.close()
        self.csv_logger.close()
        self._sync_artifacts(force=True)
        self.logger.close()
        print("✓ Training Completed Successfully!")
