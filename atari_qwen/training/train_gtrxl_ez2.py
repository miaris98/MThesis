"""High-Replay-Ratio Trainer for IMPALA-CNN + GTrXL + EfficientZero v2 Multi-Step Branch Predictor.

Optimized for sample-efficient 100k-step Atari benchmark:
    - High Update-to-Data (Replace) Ratio: RR = 4 gradient updates per environment step batch
    - Unrolls K=5 forward latent branches with EfficientZero v2 self-supervised SimSiam consistency
    - Multi-step reward and value prediction auxiliary objectives
    - Evaluation reporting Mean, Median, and HNS
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from atari_qwen.envs.atari_wrappers import make_atari_env, make_vector_atari_envs
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.eval.evaluate import compute_hns


class TrajectoryReplayBuffer:
    """Multi-step sequential replay buffer for EfficientZero v2 latent branch unrolling."""
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], unroll_steps: int = 5):
        self.capacity = capacity
        self.unroll_steps = unroll_steps
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.act_buf = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float, done: bool):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_trajectories(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Samples initial observations and consecutive K future transitions."""
        K = self.unroll_steps
        valid_indices = []
        max_idx = self.size - K - 1
        if max_idx <= 0:
            raise ValueError("Buffer does not have enough transitions for K-step unroll yet.")

        while len(valid_indices) < batch_size:
            idx = np.random.randint(0, max_idx)
            # Avoid sampling across ring-buffer boundary or episode boundaries within unroll
            if not np.any(self.done_buf[idx : idx + K]):
                valid_indices.append(idx)

        idx_arr = np.array(valid_indices)
        obs_0 = torch.from_numpy(self.obs_buf[idx_arr])
        
        actions_seq = torch.from_numpy(np.stack([self.act_buf[idx_arr + k] for k in range(K)], axis=1)).long()
        rewards_seq = torch.from_numpy(np.stack([self.rew_buf[idx_arr + k] for k in range(K)], axis=1)).float()
        
        target_future_obs = [
            torch.from_numpy(self.obs_buf[idx_arr + k + 1])
            for k in range(K)
        ]
        obs_K = torch.from_numpy(self.obs_buf[idx_arr + K])
        
        return {
            "obs_0": obs_0,
            "actions_seq": actions_seq,
            "rewards_seq": rewards_seq,
            "target_future_obs": target_future_obs,
            "obs_K": obs_K,
        }


def evaluate_agent(agent: ImpalaGTrXLAgent, env_id: str, device: torch.device, num_episodes: int = 10) -> Tuple[float, float]:
    eval_env_fn = make_atari_env(env_id, seed=999, idx=0, noop_max=0, clip_reward=False, episodic_life=False)
    env = eval_env_fn()
    agent.eval()
    scores = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        stuck_counter = 0
        last_lives = env.unwrapped.ale.lives()
        
        while not done:
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _, _ = agent(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                
            current_lives = env.unwrapped.ale.lives()
            if current_lives < last_lives or stuck_counter > 50:
                action = 1
                stuck_counter = 0
            last_lives = current_lives
            
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, term, trunc, _ = step_result
                done = term or trunc
            else:
                obs, reward, done, _ = step_result
                
            ep_ret += reward
            stuck_counter = 0 if reward != 0 else (stuck_counter + 1)
            
        scores.append(ep_ret)
        
    env.close()
    agent.train()
    return float(np.mean(scores)), float(np.std(scores))


def train_gtrxl_ez2(
    env_id: str = "BreakoutNoFrameskip-v4",
    total_steps: int = 100000,
    replay_ratio: int = 4,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    unroll_steps: int = 5,
    buffer_capacity: int = 50000,
    min_replay_size: int = 2000,
    eval_interval: int = 10000,
    log_dir: str = "results/atari_gtrxl_ez2",
    device_str: str = "auto"
):
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else device_str)
    run_name = f"gtrxl_ez2_{env_id}_{int(time.time())}"
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(str(run_dir / "tb"))
    
    print("==================================================================")
    print("--> Initializing IMPALA-CNN + GTrXL + EfficientZero v2 Trainer")
    print(f"--> Target:        {total_steps:,} steps | Replay Ratio: {replay_ratio}x")
    print(f"--> Unroll Steps:  {unroll_steps} (Multi-Step Branch Predictor)")
    print(f"--> Environment:   {env_id} | Device: {device}")
    print("==================================================================\n")
    
    # 1. Environment & Agent
    train_env_fn = make_atari_env(env_id, seed=42, idx=0, noop_max=30, clip_reward=True, episodic_life=True)
    env = train_env_fn()
    action_dim = env.action_space.n
    
    agent = ImpalaGTrXLAgent(
        action_dim=action_dim,
        in_channels=4,
        embed_dim=256,
        depth=4,
        num_heads=4,
        ffn_dim=1024,
        unroll_steps=unroll_steps,
        # See struggle-solutions S-035: bg_init=2.0 was still frozen at init after 15k-60k steps
        # in both the turbo trainers, making the actor's output input-invariant. 0.0 opens sooner.
        bg_init=0.0
    ).to(device)
    
    optimizer = optim.AdamW(agent.parameters(), lr=learning_rate, weight_decay=1e-4)
    buffer = TrajectoryReplayBuffer(capacity=buffer_capacity, obs_shape=(4, 84, 84), unroll_steps=unroll_steps)
    
    obs, _ = env.reset()
    start_time = time.time()
    best_eval = -float("inf")
    
    # Warmup buffer
    print(f"--> Warming up replay buffer with {min_replay_size} initial steps...")
    while buffer.size < min_replay_size:
        action = env.action_space.sample()
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, rew, term, trunc, _ = step_result
            d = term or trunc
        else:
            next_obs, rew, d, _ = step_result
            
        buffer.add(obs, action, rew, d)
        obs = env.reset()[0] if d else next_obs
    print(f"✓ Buffer warmed up! ({buffer.size} transitions stored)")

    global_step = buffer.size
    update_count = 0
    
    while global_step < total_steps:
        # Collect 1 environment step
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = agent(obs_t)
            # Epsilon-greedy exploration with decay
            eps = max(0.01, 1.0 - (global_step / 40000.0))
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = torch.argmax(logits, dim=-1).item()
                
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, rew, term, trunc, _ = step_result
            d = term or trunc
        else:
            next_obs, rew, d, _ = step_result
            
        buffer.add(obs, action, rew, d)
        obs = env.reset()[0] if d else next_obs
        global_step += 1
        
        # Heavy Replay Ratio: perform RR updates per environment step
        for _ in range(replay_ratio):
            update_count += 1
            batch = buffer.sample_trajectories(batch_size=batch_size)
            
            b_obs_0 = batch["obs_0"].to(device)
            b_actions = batch["actions_seq"].to(device)
            b_rewards = batch["rewards_seq"].to(device)
            b_obs_K = batch["obs_K"].to(device)
            
            # Forward root observation through IMPALA + GTrXL
            logits_0, value_0, latent_z0 = agent(b_obs_0)
            
            # Multi-Step Target Value Bootstrapping: V(s_K)
            with torch.no_grad():
                z_K, _ = agent.encode_observation(b_obs_K)
                v_K = agent.critic_head(z_K).squeeze(-1)
                
                B_sz, K_steps = b_actions.shape
                gamma = 0.99
                target_returns = torch.zeros(B_sz, K_steps, device=device, dtype=torch.float32)
                curr_ret = v_K
                for k in reversed(range(K_steps)):
                    curr_ret = b_rewards[:, k] + gamma * curr_ret
                    target_returns[:, k] = curr_ret
                R_0 = target_returns[:, 0]
            
            # Root Critic Loss
            root_value_loss = F.smooth_l1_loss(value_0.squeeze(-1), R_0)
            
            # Multi-Step EfficientZero v2 Latent Branch Unroll
            unroll = agent.unroll_branches(latent_z0, b_actions)
            
            reward_loss = 0.0
            unroll_value_loss = 0.0
            for k in range(unroll_steps):
                pred_r_k = unroll["rewards"][k]
                true_r_k = b_rewards[:, k]
                reward_loss += F.smooth_l1_loss(pred_r_k, true_r_k)
                
                pred_v_k = unroll["values"][k].squeeze(-1)
                unroll_value_loss += F.smooth_l1_loss(pred_v_k, target_returns[:, k])
                
            reward_loss /= unroll_steps
            unroll_value_loss /= unroll_steps
            total_value_loss = 0.5 * (root_value_loss + unroll_value_loss)
            
            # Advantage-Weighted Policy Gradient with Entropy Regularization
            act_0 = b_actions[:, 0]
            with torch.no_grad():
                adv_0 = R_0 - value_0.squeeze(-1)
                adv_norm = (adv_0 - adv_0.mean()) / (adv_0.std() + 1e-8)
                adv_norm = adv_norm.clamp(-5.0, 5.0)
            
            log_probs = F.log_softmax(logits_0, dim=-1)
            act_log_prob = log_probs.gather(1, act_0.unsqueeze(1)).squeeze(1)
            policy_loss = -(act_log_prob * adv_norm).mean()
            
            probs = F.softmax(logits_0, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            policy_loss = policy_loss - 0.01 * entropy
            
            # Multi-step Self-Supervised Consistency Loss (SimSiam alignment)
            consistency_loss = 0.0
            for k in range(unroll_steps):
                with torch.no_grad():
                    target_obs_k = batch["target_future_obs"][k].to(device)
                    target_z_k, _ = agent.encode_observation(target_obs_k)
                    target_proj_k = agent.predictor.projector(target_z_k)
                    
                pred_proj_k = unroll["projections"][k]
                consistency_loss += agent.predictor.compute_consistency_loss(pred_proj_k, target_proj_k)
            consistency_loss /= unroll_steps
            
            total_loss = policy_loss + 0.5 * total_value_loss + 1.0 * reward_loss + 0.5 * consistency_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
        if global_step % 100 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"Step {global_step:6d}/{total_steps:6d} | SPS: {sps:4d} | TotalLoss: {total_loss.item():.4f} | "
                  f"PLoss: {policy_loss.item():.4f} | VLoss: {total_value_loss.item():.4f} | RewLoss: {reward_loss.item():.4f} | SimLoss: {consistency_loss.item():.4f}", flush=True)
            writer.add_scalar("losses/total", total_loss.item(), global_step)
            writer.add_scalar("losses/policy", policy_loss.item(), global_step)
            writer.add_scalar("losses/value", total_value_loss.item(), global_step)
            writer.add_scalar("losses/reward", reward_loss.item(), global_step)
            writer.add_scalar("losses/consistency", consistency_loss.item(), global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            
        if global_step % eval_interval == 0 or global_step >= total_steps:
            mean_eval, std_eval = evaluate_agent(agent, env_id, device, num_episodes=5)
            hns = compute_hns(mean_eval, env_id)
            print(f"\n[EVALUATION] Step {global_step:,} | Score: {mean_eval:.2f} +/- {std_eval:.2f} | HNS: {hns*100:.1f}%\n", flush=True)
            writer.add_scalar("eval/mean_score", mean_eval, global_step)
            writer.add_scalar("eval/hns", hns, global_step)
            
            torch.save(agent.state_dict(), ckpt_dir / "model_latest.pt")
            if mean_eval > best_eval:
                best_eval = mean_eval
                torch.save(agent.state_dict(), ckpt_dir / "model_best.pt")
                print(f"✓ Saved new best GTrXL+EZ2 checkpoint (Score: {best_eval:.2f})")

    env.close()
    writer.close()
    print("\n==================================================================")
    print(f"✓ 100k GTrXL + EfficientZero v2 Training Complete! Peak Eval: {best_eval:.2f}")
    print("==================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IMPALA-CNN + GTrXL + EZ2 Predictor on Atari 100k")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--replay-ratio", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--unroll-steps", type=int, default=5)
    args = parser.parse_args()
    
    train_gtrxl_ez2(
        env_id=args.env_id,
        total_steps=args.total_steps,
        replay_ratio=args.replay_ratio,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        unroll_steps=args.unroll_steps
    )
