"""Optuna Bayesian Hyperparameter Optimization for IMPALA-CNN + GTrXL + EfficientZero v2.

Searches:
    - learning_rate: [1e-4, 1e-3]
    - replay_ratio: [2, 4]
    - unroll_steps: [3, 5]
    - bg_init: [1.0, 2.0, 3.0] (trainable GRU skip gate initialization)
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
    num_envs: int = 16,
    batch_size: int = 256
) -> float:
    # 1. Hyperparameter Search Space
    lr = trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True)
    replay_ratio = trial.suggest_categorical("replay_ratio", [2, 3])
    unroll_steps = trial.suggest_categorical("unroll_steps", [3, 5])
    bg_init = trial.suggest_categorical("bg_init", [1.0, 2.0, 3.0])
    depth = trial.suggest_categorical("depth", [2, 4])
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    consistency_weight = trial.suggest_float("consistency_weight", 0.25, 1.0)
    reward_weight = trial.suggest_float("reward_weight", 0.5, 2.0)

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
    buffer = VectorizedTrajectoryReplayBuffer(capacity=60000, obs_shape=(4, 84, 84), unroll_steps=unroll_steps)

    obs, _ = envs.reset()
    
    # Warmup
    while buffer.size < 2000:
        actions = np.random.randint(0, action_dim, size=(num_envs,))
        step_result = envs.step(actions)
        if len(step_result) == 5:
            next_obs, rews, terms, truncs, _ = step_result
            dones = np.logical_or(terms, truncs)
        else:
            next_obs, rews, dones, _ = step_result
        buffer.add_batch(obs, actions, rews, dones)
        obs = next_obs

    global_step = buffer.size
    best_eval = -float("inf")
    eval_freq = 10000

    while global_step < total_steps_per_trial:
        obs_t = torch.as_tensor(obs, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            logits, _, _ = agent(obs_t)
            eps = max(0.01, 1.0 - (global_step / 30000.0))
            if np.random.rand() < eps:
                actions = np.random.randint(0, action_dim, size=(num_envs,))
            else:
                actions = torch.argmax(logits, dim=-1).cpu().numpy()

        step_result = envs.step(actions)
        if len(step_result) == 5:
            next_obs, rews, terms, truncs, _ = step_result
            dones = np.logical_or(terms, truncs)
        else:
            next_obs, rews, dones, _ = step_result
            
        buffer.add_batch(obs, actions, rews, dones)
        obs = next_obs
        global_step += num_envs

        for _ in range(replay_ratio):
            batch = buffer.sample_trajectories(batch_size=batch_size)
            b_obs_0 = batch["obs_0"].to(device)
            b_actions = batch["actions_seq"].to(device)
            b_rewards = batch["rewards_seq"].to(device)
            target_future = batch["target_future_obs"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits_0, value_0, latent_z0 = agent(b_obs_0)
                policy_loss = F.cross_entropy(logits_0, b_actions[:, 0])
                unroll = agent.unroll_branches(latent_z0, b_actions)

                reward_loss = 0.0
                for k in range(unroll_steps):
                    reward_loss += F.smooth_l1_loss(unroll["rewards"][k], b_rewards[:, k])
                reward_loss /= unroll_steps

                B, K, C, H, W = target_future.shape
                flat_targets = target_future.view(B * K, C, H, W)
                with torch.no_grad():
                    flat_target_z, _ = agent.encode_observation(flat_targets)
                    flat_target_proj = agent.predictor.projector(flat_target_z)
                    target_projs = flat_target_proj.view(B, K, -1)

                consistency_loss = 0.0
                for k in range(unroll_steps):
                    consistency_loss += agent.predictor.compute_consistency_loss(unroll["projections"][k], target_projs[:, k])
                consistency_loss /= unroll_steps

                total_loss = policy_loss + reward_weight * reward_loss + consistency_weight * consistency_loss

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
    parser.add_argument("--steps-per-trial", type=int, default=50000)
    parser.add_argument("--storage", type=str, default="sqlite:///results/atari_gtrxl_optuna.db")
    parser.add_argument("--study-name", type=str, default="gtrxl_ez2_hparam_study")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=10000, interval_steps=10000)

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
