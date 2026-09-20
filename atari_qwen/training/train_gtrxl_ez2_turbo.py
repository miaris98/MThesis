"""Vectorized High-Throughput Turbo Trainer for IMPALA-CNN + GTrXL + EfficientZero v2.

Key Speed & Memory Enhancements:
    1. Vectorized Atari Environments (num_envs = 16): eliminates single-core CPU stepping bottleneck.
    2. Large Batch Processing (batch_size = 256): saturates RTX 4070 Ti Super tensor cores (utilizing ~8-10 GB VRAM).
    3. Batched Multi-Step Target Projection: encodes all B x K future observations in a single CUDA kernel pass.
    4. Mixed Precision (torch.amp.autocast with bfloat16): 3x faster attention and convolution throughput.
    5. Prioritized sequential trajectory storage with thread-safe additions.
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

from atari_qwen.envs.atari_wrappers import make_atari_env, make_vector_atari_envs, StickyActionEnv
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.eval.evaluate import compute_hns


class VectorizedTrajectoryReplayBuffer:
    """High-throughput circular buffer supporting vectorized multi-step trajectory sampling."""
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], unroll_steps: int = 5):
        self.capacity = capacity
        self.unroll_steps = unroll_steps
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.act_buf = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.bool_)
        # log-probability of the action actually taken, under the behavior policy active at
        # collection time (the eps-greedy/sampling mixture) -- needed for PPO-style importance
        # sampling correction against off-policy replay drift (see struggle-solutions S-025/S-026).
        self.logp_buf = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        rew_batch: np.ndarray,
        done_batch: np.ndarray,
        logp_batch: np.ndarray
    ):
        num_items = len(act_batch)
        for i in range(num_items):
            self.obs_buf[self.ptr] = obs_batch[i]
            self.act_buf[self.ptr] = act_batch[i]
            self.rew_buf[self.ptr] = rew_batch[i]
            self.done_buf[self.ptr] = done_batch[i]
            self.logp_buf[self.ptr] = logp_batch[i]

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample_trajectories(self, batch_size: int) -> Dict[str, torch.Tensor]:
        K = self.unroll_steps
        max_idx = self.size - K - 1
        if max_idx <= 0:
            raise ValueError("Buffer does not have enough transitions for K-step unroll yet.")

        valid_indices = []
        while len(valid_indices) < batch_size:
            idx = np.random.randint(0, max_idx)
            if not np.any(self.done_buf[idx : idx + K]):
                valid_indices.append(idx)

        idx_arr = np.array(valid_indices)
        obs_0 = torch.from_numpy(self.obs_buf[idx_arr])

        actions_seq = torch.from_numpy(np.stack([self.act_buf[idx_arr + k] for k in range(K)], axis=1)).long()
        rewards_seq = torch.from_numpy(np.stack([self.rew_buf[idx_arr + k] for k in range(K)], axis=1)).float()
        logp_0 = torch.from_numpy(self.logp_buf[idx_arr]).float()

        # Stack all future targets into a single tensor for 1-pass parallel GPU batch encoding: (B, K, C, H, W)
        future_obs_list = [self.obs_buf[idx_arr + k + 1] for k in range(K)]
        target_future_obs = torch.from_numpy(np.stack(future_obs_list, axis=1))  # (B, K, 4, 84, 84)
        obs_K = torch.from_numpy(self.obs_buf[idx_arr + K])  # (B, 4, 84, 84)

        return {
            "obs_0": obs_0,
            "actions_seq": actions_seq,
            "rewards_seq": rewards_seq,
            "target_future_obs": target_future_obs,
            "obs_K": obs_K,
            "logp_0": logp_0,
        }


def evaluate_agent(
    agent: ImpalaGTrXLAgent,
    env_id: str,
    device: torch.device,
    num_episodes: int = 5,
    use_lookahead: bool = False,
    # E35: sample from softmax(logits / temperature) instead of deterministic argmax. Under
    # argmax, a near-zero (~0.0002) logit gap between actions still locks in a 100%-constant
    # action; temperature > 0 lets the eval score actually reflect the logit spread. 0 keeps
    # the original deterministic-argmax behavior.
    eval_temperature: float = 0.0,
    # E36: with probability sticky_action_p, repeat the previous action rather than the
    # requested one (Machado et al. sticky actions) -- breaks the deterministic-emulator
    # looping a constant-action policy exploits to reproduce the same score every time.
    sticky_action_p: float = 0.0,
) -> Tuple[float, float]:
    eval_env_fn = make_atari_env(env_id, seed=999, idx=0, noop_max=0, clip_reward=False, episodic_life=False)
    env = eval_env_fn()
    if sticky_action_p > 0.0:
        env = StickyActionEnv(env, p=sticky_action_p)
    agent.eval()
    scores = []
    action_counts: Dict[int, int] = {}
    action_meanings = env.unwrapped.get_action_meanings()

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        stuck_counter = 0
        last_lives = env.unwrapped.ale.lives()

        # Initial FIRE on reset for games like Breakout that require it to launch the ball
        has_fire = "FIRE" in action_meanings and len(action_meanings) >= 2
        if has_fire:
            step_result = env.step(1)
            action_counts[1] = action_counts.get(1, 0) + 1
            if len(step_result) == 5:
                obs, r, term, trunc, _ = step_result
                done = term or trunc
            else:
                obs, r, done, _ = step_result
            ep_ret += r

        while not done:
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                if use_lookahead:
                    q_vals, _ = agent.evaluate_actions_lookahead(obs_t)
                    logits, _, _ = agent(obs_t)
                    action = torch.argmax(logits + 0.5 * q_vals, dim=-1).item()
                elif eval_temperature > 0.0:
                    logits, _, _ = agent(obs_t)
                    probs = F.softmax(logits.float() / eval_temperature, dim=-1)
                    action = torch.distributions.Categorical(probs=probs).sample().item()
                else:
                    logits, _, _ = agent(obs_t)
                    action = torch.argmax(logits, dim=-1).item()

            current_lives = env.unwrapped.ale.lives()
            if has_fire and (current_lives < last_lives or stuck_counter > 30):
                action = 1
                stuck_counter = 0
            last_lives = current_lives

            action_counts[action] = action_counts.get(action, 0) + 1
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

    total_actions = sum(action_counts.values())
    if total_actions > 0:
        hist_str = ", ".join(
            f"{action_meanings[a] if a < len(action_meanings) else a}:{action_counts.get(a, 0)}"
            f" ({100.0 * action_counts.get(a, 0) / total_actions:.0f}%)"
            for a in sorted(action_counts)
        )
        print(f"[EVAL ACTIONS] {hist_str}", flush=True)

    return float(np.mean(scores)), float(np.std(scores))


def train_turbo_gtrxl_ez2(
    env_id: str = "BreakoutNoFrameskip-v4",
    total_steps: int = 100000,
    num_envs: int = 32,
    replay_ratio: int = 2,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    unroll_steps: int = 5,
    # Kept well below typical total_steps so the buffer actually cycles and batches stay
    # recent (see struggle-solutions S-028 -- 100k never wrapped within a 60k-step run,
    # so widening num_envs alone didn't reduce effective staleness as intended).
    buffer_capacity: int = 15000,
    min_replay_size: int = 2000,
    eval_interval: int = 10000,
    log_dir: str = "results/atari_gtrxl_ez2",
    device_str: str = "auto"
):
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else device_str)
    run_name = f"gtrxl_ez2_turbo_{env_id}_{int(time.time())}"
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(str(run_dir / "tb"))
    
    print("==================================================================")
    print("--> Initializing Turbo Vectorized IMPALA + GTrXL + EfficientZero v2")
    print(f"--> Target:        {total_steps:,} steps | Vector Envs: {num_envs}")
    print(f"--> Batch Size:    {batch_size} (Tensor-Core Saturated)")
    print(f"--> Replay Ratio:  {replay_ratio}x | Unroll Steps: {unroll_steps}")
    print(f"--> Environment:   {env_id} | Device: {device}")
    print("==================================================================\n")
    
    # 1. Vectorized Environments & Agent
    envs = make_vector_atari_envs(env_id, num_envs=num_envs, seed=42, noop_max=30, clip_reward=True, episodic_life=True)
    action_dim = envs.single_action_space.n
    
    agent = ImpalaGTrXLAgent(
        action_dim=action_dim,
        in_channels=4,
        embed_dim=256,
        depth=4,
        num_heads=4,
        ffn_dim=1024,
        unroll_steps=unroll_steps,
        # See struggle-solutions S-035: bg_init=2.0 was still frozen at init after a full 60k-step
        # run, making the actor's output input-invariant. 0.0 lets gates open much sooner.
        bg_init=0.0
    ).to(device)
    
    optimizer = optim.AdamW(agent.parameters(), lr=learning_rate, weight_decay=1e-4)
    buffer = VectorizedTrajectoryReplayBuffer(capacity=buffer_capacity, obs_shape=(4, 84, 84), unroll_steps=unroll_steps)
    
    obs, _ = envs.reset()
    start_time = time.time()
    best_eval = -float("inf")
    
    # Warmup buffer using vectorized rollouts
    print(f"--> Warming up replay buffer with {min_replay_size} initial steps...")
    uniform_logp = np.full((num_envs,), -np.log(action_dim), dtype=np.float32)
    while buffer.size < min_replay_size:
        actions = np.random.randint(0, action_dim, size=(num_envs,))
        step_result = envs.step(actions)
        if len(step_result) == 5:
            next_obs, rews, terms, truncs, _ = step_result
            dones = np.logical_or(terms, truncs)
        else:
            next_obs, rews, dones, _ = step_result

        # Warmup actions are drawn purely uniform-random, so their true behavior-policy
        # probability is exactly 1/action_dim -- no model forward needed to know it.
        buffer.add_batch(obs, actions, rews, dones, uniform_logp)
        obs = next_obs
    print(f"✓ Buffer warmed up! ({buffer.size:,} transitions stored)")

    global_step = buffer.size
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    while global_step < total_steps:
        # 1. Vectorized Environment Step (16 parallel game frames)
        obs_t = torch.as_tensor(obs, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _, _ = agent(obs_t)
            probs = F.softmax(logits.float(), dim=-1)
            eps = max(0.1, 1.0 - (global_step / 100000.0))
            # Per-env independent eps-greedy coin flip (see struggle-solutions S-033): a single
            # shared `np.random.rand() < eps` check applied one coin toss to the WHOLE vectorized
            # batch, so every step was either all-32-random or all-32-on-policy. That correlates
            # exploration noise across the batch (bursty blocks of pure-random vs. pure-on-policy
            # transitions) instead of the per-env diversity eps-greedy is meant to provide -- flip
            # independently per env instead.
            random_mask = torch.rand(num_envs, device=device) < eps
            random_actions = torch.randint(0, action_dim, (num_envs,), device=device)
            # Sample from the policy's own distribution (not argmax) so the collected
            # action actually reflects the stochastic policy being importance-corrected below.
            policy_actions = torch.distributions.Categorical(probs=probs).sample()
            actions_t = torch.where(random_mask, random_actions, policy_actions)

            # True behavior-policy probability of the taken action under the eps-greedy mixture
            # b(a|s) = eps/|A| + (1-eps)*policy(a|s), regardless of which branch produced it --
            # this is what PPO-style importance sampling needs as the "old" reference probability.
            policy_prob_of_action = probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            behavior_prob = eps / action_dim + (1.0 - eps) * policy_prob_of_action
            behavior_logp = torch.log(behavior_prob.clamp(min=1e-8))

            actions = actions_t.cpu().numpy()
            logp_step = behavior_logp.cpu().numpy()

        step_result = envs.step(actions)
        if len(step_result) == 5:
            next_obs, rews, terms, truncs, _ = step_result
            dones = np.logical_or(terms, truncs)
        else:
            next_obs, rews, dones, _ = step_result

        buffer.add_batch(obs, actions, rews, dones, logp_step)
        obs = next_obs
        global_step += num_envs
        
        # 2. Fast Batched Multi-Step RL & Dynamics Gradient Updates
        for _ in range(replay_ratio):
            batch = buffer.sample_trajectories(batch_size=batch_size)
            
            b_obs_0 = batch["obs_0"].to(device)
            b_actions = batch["actions_seq"].to(device)
            b_rewards = batch["rewards_seq"].to(device)
            target_future = batch["target_future_obs"].to(device)  # (B, K, 4, 84, 84) -- s_1..s_K
            b_logp_old = batch["logp_0"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                # 1. Forward root observation through IMPALA + GTrXL
                logits_0, value_0, latent_z0 = agent(b_obs_0)  # value_0: (B, 1)

                # 2. Encode the K real future observations ONCE -- reused below both for the
                # GAE value targets and the SimSiam consistency-loss targets (previously two
                # separate encode_observation passes; target_future[:, K-1] is exactly s_K, so
                # no separate "obs_K" encoding is needed either).
                B_sz, K_steps = b_actions.shape
                flat_targets = target_future.reshape(B_sz * K_steps, *target_future.shape[2:])
                with torch.no_grad():
                    flat_target_z, _ = agent.encode_observation(flat_targets)
                    flat_target_proj = agent.predictor.projector(flat_target_z)
                    target_projs = flat_target_proj.view(B_sz, K_steps, -1)
                    future_values = agent.critic_head(flat_target_z).squeeze(-1).view(B_sz, K_steps)  # V(s_1..s_K)

                # 3. Multi-Step GAE(lambda) Advantage/Return Targets, using these real encoded
                # values rather than a single n-step bootstrap sum. A raw bootstrap difference is
                # high-variance early in training when the critic itself is unreliable -- exactly
                # the noisy-advantage failure mode behind the repeated collapse in struggle-
                # solutions S-025/S-026/S-027/S-028. GAE blending many n-step estimators together
                # is the mechanism the successfully-trained on-policy comparison run (S-029) relies
                # on; this brings the same idea into the replay-buffer's K-step windows without
                # requiring a full on-policy rollout.
                with torch.no_grad():
                    gamma = 0.99
                    gae_lambda = 0.95
                    v_root = value_0.squeeze(-1)  # V(s_0)
                    values_seq = torch.cat([v_root.unsqueeze(1), future_values], dim=1)  # V(s_0..s_K)

                    advantages = torch.zeros(B_sz, K_steps, device=device, dtype=torch.float32)
                    gae = torch.zeros(B_sz, device=device, dtype=torch.float32)
                    for k in reversed(range(K_steps)):
                        delta = b_rewards[:, k] + gamma * values_seq[:, k + 1] - values_seq[:, k]
                        gae = delta + gamma * gae_lambda * gae
                        advantages[:, k] = gae
                    target_returns = advantages + values_seq[:, :K_steps]
                    R_0 = target_returns[:, 0]
                    adv_0 = advantages[:, 0]

                # 4. Root Critic Loss
                root_value_loss = F.smooth_l1_loss(value_0.squeeze(-1), R_0)

                # 5. Multi-Step EfficientZero v2 Latent Branch Unroll
                unroll = agent.unroll_branches(latent_z0, b_actions)

                reward_loss = 0.0
                unroll_value_loss = 0.0
                consistency_loss = 0.0
                for k in range(unroll_steps):
                    pred_r_k = unroll["rewards"][k]
                    true_r_k = b_rewards[:, k]
                    reward_loss += F.smooth_l1_loss(pred_r_k, true_r_k)

                    pred_v_k = unroll["values"][k].squeeze(-1)
                    target_v_k = target_returns[:, k]
                    unroll_value_loss += F.smooth_l1_loss(pred_v_k, target_v_k)

                    pred_proj_k = unroll["projections"][k]
                    target_proj_k = target_projs[:, k]
                    consistency_loss += agent.predictor.compute_consistency_loss(pred_proj_k, target_proj_k)

                reward_loss /= unroll_steps
                unroll_value_loss /= unroll_steps
                consistency_loss /= unroll_steps
                total_value_loss = 0.5 * (root_value_loss + unroll_value_loss)

                # 6. PPO-Clipped Advantage-Weighted Policy Gradient with Entropy Regularization
                #
                # Replay-buffer samples are off-policy by construction (collected under earlier,
                # stale versions of the policy), but a plain REINFORCE/advantage-actor-critic update
                # assumes on-policy data -- with no correction, a sample the current policy now
                # considers very unlikely produces a huge-magnitude log-prob that dominates the batch
                # gradient and drives the policy to collapse onto a single action (struggle-solutions
                # S-024/S-025/S-026). PPO's clipped importance-sampling surrogate bounds the update
                # from any one sample regardless of how far the policy has drifted since collection.
                act_0 = b_actions[:, 0]
                adv_norm = (adv_0 - adv_0.mean()) / (adv_0.std() + 1e-8)
                adv_norm = adv_norm.clamp(-5.0, 5.0)

                log_probs = F.log_softmax(logits_0, dim=-1)
                act_log_prob = log_probs.gather(1, act_0.unsqueeze(1)).squeeze(1)

                # exp() of a huge log-ratio can overflow to inf even though the clipped surrogate
                # would discard it anyway -- clamp the exponent, not just the ratio, to stay finite.
                log_ratio = (act_log_prob - b_logp_old).clamp(-20.0, 20.0)
                ratio = log_ratio.exp()
                clip_eps = 0.2
                surr1 = ratio * adv_norm
                surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
                policy_loss = -torch.min(surr1, surr2).mean()

                probs = F.softmax(logits_0, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                policy_loss = policy_loss - 0.04 * entropy

                total_loss = policy_loss + 0.5 * total_value_loss + 1.0 * reward_loss + 0.5 * consistency_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            
        if global_step % 500 < num_envs:
            sps = int(global_step / (time.time() - start_time))
            print(f"Step {global_step:6d}/{total_steps:6d} | SPS: {sps:4d} | TotalLoss: {total_loss.item():.4f} | "
                  f"PLoss: {policy_loss.item():.4f} | VLoss: {total_value_loss.item():.4f} | RewLoss: {reward_loss.item():.4f} | SimLoss: {consistency_loss.item():.4f}", flush=True)
            writer.add_scalar("losses/total", total_loss.item(), global_step)
            writer.add_scalar("losses/policy", policy_loss.item(), global_step)
            writer.add_scalar("losses/value", total_value_loss.item(), global_step)
            writer.add_scalar("losses/reward", reward_loss.item(), global_step)
            writer.add_scalar("losses/consistency", consistency_loss.item(), global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            
        if global_step % eval_interval < num_envs or global_step >= total_steps:
            mean_eval, std_eval = evaluate_agent(agent, env_id, device, num_episodes=5)
            hns = compute_hns(mean_eval, env_id)
            # compute_hns() already returns a percentage; don't multiply by 100 again (was a
            # 100x display bug -- e.g. a real 32.29% printed as "3229.2%").
            print(f"\n[EVALUATION] Step {global_step:,} | Score: {mean_eval:.2f} +/- {std_eval:.2f} | HNS: {hns:.1f}%\n", flush=True)
            writer.add_scalar("eval/mean_score", mean_eval, global_step)
            writer.add_scalar("eval/hns", hns, global_step)
            
            torch.save(agent.state_dict(), ckpt_dir / "model_latest.pt")
            if mean_eval > best_eval:
                best_eval = mean_eval
                torch.save(agent.state_dict(), ckpt_dir / "model_best.pt")
                print(f"✓ Saved new best GTrXL+EZ2 checkpoint (Score: {best_eval:.2f})", flush=True)

    envs.close()
    writer.close()
    print("\n==================================================================")
    print(f"✓ Turbo GTrXL + EfficientZero v2 Complete! Peak Eval: {best_eval:.2f}")
    print("==================================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turbo Vectorized Trainer for IMPALA + GTrXL + EZ2")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--replay-ratio", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--unroll-steps", type=int, default=5)
    args = parser.parse_args()
    
    train_turbo_gtrxl_ez2(
        env_id=args.env_id,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        replay_ratio=args.replay_ratio,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        unroll_steps=args.unroll_steps
    )
