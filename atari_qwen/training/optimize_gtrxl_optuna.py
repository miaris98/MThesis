"""Optuna Bayesian Hyperparameter Optimization for IMPALA-CNN + GTrXL + EfficientZero v2.

Searches:
    - learning_rate: [1e-4, 1e-3]
    - replay_ratio: [2, 4]
    - unroll_steps: [3, 5]
    - bg_init: [-1.0, 0.0, 1.0] (trainable GRU skip gate initialization -- see struggle-solutions
      S-035: the old [1.0, 2.0, 3.0] range was measured to keep every gate 73-95% closed, and
      every prior trial in this study left the gates frozen near their init after 50k steps,
      making the actor's output input-invariant regardless of any other hyperparameter)
    - consistency_loss_weight: [0.25, 0.5, 1.0]
    - reward_loss_weight: [0.5, 1.0, 2.0]
    - depth: [2, 4] (GTrXL layers)
    - num_heads: [2, 4]
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from atari_qwen.envs.atari_wrappers import make_vector_atari_envs, make_atari_env
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.training.train_gtrxl_ez2_turbo import VectorizedTrajectoryReplayBuffer, evaluate_agent


def objective(
    trial: optuna.Trial,
    env_id: str = "BreakoutNoFrameskip-v4",
    total_steps_per_trial: int = 50000,
    num_envs: int = 32,
    batch_size: int = 256
) -> float:
    # 1. Hyperparameter Search Space
    lr = trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True)
    replay_ratio = trial.suggest_categorical("replay_ratio", [2, 3])
    unroll_steps = trial.suggest_categorical("unroll_steps", [3, 5])
    bg_init = trial.suggest_categorical("bg_init", [-1.0, 0.0, 1.0])
    depth = trial.suggest_categorical("depth", [2, 4])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    value_weight = trial.suggest_float("value_weight", 0.25, 1.0)
    consistency_weight = trial.suggest_float("consistency_weight", 0.25, 1.0)
    reward_weight = trial.suggest_float("reward_weight", 0.5, 2.0)
    ent_coef = trial.suggest_float("ent_coef", 0.005, 0.05, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # 2. Environments & Model
    envs = make_vector_atari_envs(env_id, num_envs=num_envs, seed=trial.number * 100, noop_max=30, clip_reward=True, episodic_life=True)
    action_dim = envs.single_action_space.n
    
    agent = ImpalaGTrXLAgent(
        action_dim=action_dim,
        in_channels=4,
        embed_dim=256,
        depth=depth,
        num_heads=num_heads,
        ffn_dim=1024,
        unroll_steps=unroll_steps,
        bg_init=bg_init
    ).to(device)

    optimizer = optim.AdamW(agent.parameters(), lr=lr, weight_decay=1e-4)
    # Kept well below total_steps_per_trial so the buffer actually cycles and batches stay
    # recent (see struggle-solutions S-028).
    buffer = VectorizedTrajectoryReplayBuffer(capacity=15000, obs_shape=(4, 84, 84), unroll_steps=unroll_steps)

    obs, _ = envs.reset()

    # Warmup -- actions drawn purely uniform-random, so their true behavior-policy probability
    # is exactly 1/action_dim (no model forward needed to know it).
    uniform_logp = np.full((num_envs,), -np.log(action_dim), dtype=np.float32)
    while buffer.size < 2000:
        actions = np.random.randint(0, action_dim, size=(num_envs,))
        step_result = envs.step(actions)
        if len(step_result) == 5:
            next_obs, rews, terms, truncs, _ = step_result
            dones = np.logical_or(terms, truncs)
        else:
            next_obs, rews, dones, _ = step_result
        buffer.add_batch(obs, actions, rews, dones, uniform_logp)
        obs = next_obs

    global_step = buffer.size
    best_eval = -float("inf")
    eval_freq = 10000

    while global_step < total_steps_per_trial:
        obs_t = torch.as_tensor(obs, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _, _ = agent(obs_t)
            probs = F.softmax(logits.float(), dim=-1)
            eps = max(0.01, 1.0 - (global_step / 70000.0))
            # Per-env independent eps-greedy coin flip (see struggle-solutions S-033), matching
            # the fix in train_gtrxl_ez2_turbo.py -- a single shared coin toss for the whole
            # vectorized batch correlates exploration noise across envs instead of diversifying it.
            random_mask = torch.rand(num_envs, device=device) < eps
            random_actions = torch.randint(0, action_dim, (num_envs,), device=device)
            # Sample from the policy's own distribution (not argmax) so the collected action
            # actually reflects the stochastic policy being importance-corrected below.
            policy_actions = torch.distributions.Categorical(probs=probs).sample()
            actions_t = torch.where(random_mask, random_actions, policy_actions)

            # True behavior-policy probability under the eps-greedy mixture
            # b(a|s) = eps/|A| + (1-eps)*policy(a|s) -- the "old" reference probability PPO needs.
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

        for _ in range(replay_ratio):
            batch = buffer.sample_trajectories(batch_size=batch_size)
            b_obs_0 = batch["obs_0"].to(device)
            b_actions = batch["actions_seq"].to(device)
            b_rewards = batch["rewards_seq"].to(device)
            target_future = batch["target_future_obs"].to(device)  # (B, K, 4, 84, 84) -- s_1..s_K
            b_logp_old = batch["logp_0"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits_0, value_0, latent_z0 = agent(b_obs_0)

                # Encode the K real future observations ONCE -- reused below both for the GAE
                # value targets and the SimSiam consistency-loss targets (target_future[:, K-1]
                # is exactly s_K, so no separate "obs_K" encoding pass is needed either).
                B_sz, K_steps = b_actions.shape
                flat_targets = target_future.reshape(B_sz * K_steps, *target_future.shape[2:])
                with torch.no_grad():
                    flat_target_z, _ = agent.encode_observation(flat_targets)
                    flat_target_proj = agent.predictor.projector(flat_target_z)
                    target_projs = flat_target_proj.view(B_sz, K_steps, -1)
                    future_values = agent.critic_head(flat_target_z).squeeze(-1).view(B_sz, K_steps)  # V(s_1..s_K)

                # Multi-Step GAE(lambda) Advantage/Return Targets using these real encoded
                # values rather than a single n-step bootstrap sum (see struggle-solutions
                # S-029 -- this is the mechanism the successfully-trained on-policy comparison
                # run relies on, brought into the replay buffer's K-step windows).
                with torch.no_grad():
                    gamma = 0.99
                    gae_lambda = 0.95
                    v_root = value_0.squeeze(-1)
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

                root_value_loss = F.smooth_l1_loss(value_0.squeeze(-1), R_0)
                unroll = agent.unroll_branches(latent_z0, b_actions)

                reward_loss = 0.0
                unroll_value_loss = 0.0
                consistency_loss = 0.0
                for k in range(unroll_steps):
                    reward_loss += F.smooth_l1_loss(unroll["rewards"][k], b_rewards[:, k])
                    pred_v_k = unroll["values"][k].squeeze(-1)
                    unroll_value_loss += F.smooth_l1_loss(pred_v_k, target_returns[:, k])
                    consistency_loss += agent.predictor.compute_consistency_loss(unroll["projections"][k], target_projs[:, k])

                reward_loss /= unroll_steps
                unroll_value_loss /= unroll_steps
                consistency_loss /= unroll_steps
                total_value_loss = 0.5 * (root_value_loss + unroll_value_loss)

                # PPO-Clipped Advantage-Weighted Policy Gradient (see struggle-solutions
                # S-024/S-025/S-026: plain REINFORCE on off-policy replay samples with no
                # importance-sampling correction drives the policy to collapse onto a single
                # action). The clipped surrogate bounds the update from any one stale sample
                # regardless of how far the policy has drifted since it was collected.
                act_0 = b_actions[:, 0]
                adv_norm = (adv_0 - adv_0.mean()) / (adv_0.std() + 1e-8)
                adv_norm = adv_norm.clamp(-5.0, 5.0)

                log_probs = F.log_softmax(logits_0, dim=-1)
                act_log_prob = log_probs.gather(1, act_0.unsqueeze(1)).squeeze(1)

                log_ratio = (act_log_prob - b_logp_old).clamp(-20.0, 20.0)
                ratio = log_ratio.exp()
                clip_eps = 0.2
                surr1 = ratio * adv_norm
                surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
                policy_loss = -torch.min(surr1, surr2).mean()

                probs = F.softmax(logits_0, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                policy_loss = policy_loss - ent_coef * entropy

                total_loss = (
                    policy_loss
                    + value_weight * total_value_loss
                    + reward_weight * reward_loss
                    + consistency_weight * consistency_loss
                )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        # Intermediate evaluation and Optuna pruning
        if global_step % eval_freq < num_envs:
            mean_eval, _ = evaluate_agent(agent, env_id, device, num_episodes=5)
            best_eval = max(best_eval, mean_eval)
            trial.report(mean_eval, step=global_step)
            print(f"--> Trial #{trial.number} | Step {global_step:,} | Eval Score: {mean_eval:.2f}", flush=True)

            if trial.should_prune():
                envs.close()
                raise optuna.exceptions.TrialPruned()

    envs.close()
    final_score, _ = evaluate_agent(agent, env_id, device, num_episodes=10)
    print(f"✓ Trial #{trial.number} FINISHED | Final Eval Score: {final_score:.2f}", flush=True)
    return max(best_eval, final_score)


def main():
    parser = argparse.ArgumentParser(description="Optuna Study for IMPALA-CNN + GTrXL + EfficientZero v2")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--steps-per-trial", type=int, default=100000)
    parser.add_argument("--storage", type=str, default="sqlite:///results/atari_gtrxl_optuna.db")
    parser.add_argument("--study-name", type=str, default="gtrxl_ez2_hparam_study")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=40000, interval_steps=10000)

    print("==================================================================")
    print("--> Starting Bayesian Hyperparameter Study for GTrXL + EfficientZero v2")
    print(f"--> Environment: {args.env_id} | Trials: {args.n_trials}")
    print(f"--> Storage:     {args.storage}")
    print("==================================================================\n", flush=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    study.optimize(
        lambda t: objective(t, env_id=args.env_id, total_steps_per_trial=args.steps_per_trial),
        n_trials=args.n_trials,
        gc_after_trial=True
    )

    print("\n==================================================================")
    print("HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print(f"Best Trial #{study.best_trial.number}: Score = {study.best_value:.2f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:24s}: {v}")
    print("==================================================================\n", flush=True)


if __name__ == "__main__":
    main()
