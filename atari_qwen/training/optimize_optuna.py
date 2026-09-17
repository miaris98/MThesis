"""Optuna Bayesian Hyperparameter Optimization for Atari Qwen Reinforcement Learning."""
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
import torch.optim as optim

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from atari_qwen.config.atari_config import AtariConfig, get_config
from atari_qwen.envs.atari_wrappers import make_vector_atari_envs, make_atari_env
from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic
from atari_qwen.training.train_ppo import evaluate_policy


def objective(trial: "optuna.Trial", env_id: str, total_timesteps_per_trial: int = 500_000, num_envs: int = 16) -> float:
    # 1. Sample Hyperparameters
    lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.005, 0.05, log=True)
    clip_coef = trial.suggest_categorical("clip_coef", [0.1, 0.2])
    num_steps = trial.suggest_categorical("num_steps", [64, 128])
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.98)
    encoder_type = trial.suggest_categorical("encoder_type", ["nature_cnn", "impala_cnn"])
    model_preset = trial.suggest_categorical("model_preset", ["tiny", "small"])

    # 2. Config setup
    use_ckpt = (model_preset == "small")
    num_minibatches = 8 if (model_preset == "small" or num_steps == 128) else 4

    config = get_config(
        env_id=env_id,
        model_preset=model_preset,
        encoder_type=encoder_type,
        num_envs=num_envs,
        num_steps=num_steps,
        total_timesteps=total_timesteps_per_trial,
        learning_rate=lr,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        num_minibatches=num_minibatches,
        use_gradient_checkpointing=use_ckpt,
        track_mlflow=False,
        exp_name=f"optuna_trial_{trial.number}"
    )

    # 3. Device & Seeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32

    np.random.seed(config.seed + trial.number)
    torch.manual_seed(config.seed + trial.number)

    # 4. Environments & Model
    envs = make_vector_atari_envs(
        env_id=config.env_id,
        num_envs=config.num_envs,
        seed=config.seed + trial.number,
        asynchronous=True
    )
    action_dim = envs.single_action_space.n

    model = QwenAtariActorCritic(
        action_dim=action_dim,
        in_channels=config.frame_stack,
        preset=config.model_preset,
        encoder_type=config.encoder_type,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-5, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(config.use_amp and amp_dtype == torch.float16))

    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = config.total_timesteps // batch_size

    obs_buffer = torch.zeros((config.num_steps, config.num_envs, config.frame_stack, 84, 84), dtype=torch.uint8)
    actions_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long)
    logprobs_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32)
    rewards_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32)
    dones_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32)
    values_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32)

    global_step = 0
    best_eval = -float("inf")
    reset_res = envs.reset()
    next_obs = torch.as_tensor(reset_res[0] if isinstance(reset_res, tuple) else reset_res, device=device)
    next_done = torch.zeros(config.num_envs, device=device)

    eval_interval_steps = 50_000

    try:
        for update in range(1, num_updates + 1):
            # Rollout
            model.eval()
            for step in range(config.num_steps):
                global_step += config.num_envs
                obs_buffer[step] = next_obs.cpu()
                dones_buffer[step] = next_done.cpu()

                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=config.use_amp):
                        action, logprob, _, value = model.get_action_and_value(next_obs)
                    values_buffer[step] = value.squeeze(-1).float().cpu()

                actions_buffer[step] = action.cpu()
                logprobs_buffer[step] = logprob.cpu()

                step_res = envs.step(action.cpu().numpy())
                if len(step_res) == 5:
                    next_obs_np, reward, term, trunc, _ = step_res
                    done_np = np.logical_or(term, trunc)
                else:
                    next_obs_np, reward, done_np, _ = step_res

                rewards_buffer[step] = torch.as_tensor(reward, dtype=torch.float32)
                next_obs = torch.as_tensor(next_obs_np, device=device)
                next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            # GAE
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=config.use_amp):
                    next_value = model.get_value(next_obs).squeeze(-1).float().cpu()
                advantages = torch.zeros_like(rewards_buffer)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.cpu()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_buffer[t + 1]
                        nextvalues = values_buffer[t + 1]
                    delta = rewards_buffer[t] + config.gamma * nextvalues * nextnonterminal - values_buffer[t]
                    advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values_buffer

            # Optimize
            b_obs = obs_buffer.reshape((-1, config.frame_stack, 84, 84))
            b_logprobs = logprobs_buffer.reshape(-1)
            b_actions = actions_buffer.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buffer.reshape(-1)

            model.train()
            b_inds = np.arange(batch_size)
            for epoch in range(config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    mb_obs = b_obs[mb_inds].to(device)
                    mb_actions = b_actions[mb_inds].to(device)
                    mb_logprobs = b_logprobs[mb_inds].to(device)
                    mb_advantages = b_advantages[mb_inds].to(device)
                    mb_returns = b_returns[mb_inds].to(device)
                    mb_values = b_values[mb_inds].to(device)

                    if config.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=config.use_amp):
                        _, newlogprob, entropy, newvalue = model.get_action_and_value(mb_obs, mb_actions)
                        newvalue = newvalue.squeeze(-1)

                        logratio = newlogprob - mb_logprobs
                        ratio = logratio.exp()

                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                        loss = pg_loss - config.ent_coef * entropy.mean() + config.vf_coef * v_loss

                    optimizer.zero_grad()
                    if config.use_amp and amp_dtype == torch.float16:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()

            # Evaluation & Pruning check
            if global_step % eval_interval_steps < batch_size or update == num_updates:
                eval_mean, _ = evaluate_policy(
                    model=model,
                    env_id=config.env_id,
                    device=device,
                    num_episodes=5,
                    seed=config.seed + trial.number * 100
                )
                best_eval = max(best_eval, eval_mean)
                trial.report(eval_mean, step=global_step)

                if trial.should_prune():
                    raise optuna.TrialPruned()

    finally:
        envs.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_eval


def run_optuna_study(
    env_id: str = "BreakoutNoFrameskip-v4",
    n_trials: int = 20,
    storage_path: str = "sqlite:///results/atari_qwen/optuna.db",
    study_name: str = "qwen_breakout_tuning",
    total_timesteps_per_trial: int = 500_000,
    num_envs: int = 16
):
    """Run Optuna study with persistent SQLite storage and MedianPruner."""
    if not OPTUNA_AVAILABLE:
        raise SystemExit("Optuna is not installed. Run 'pip install optuna'")

    Path("results/atari_qwen").mkdir(parents=True, exist_ok=True)

    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=1)
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
        sampler=sampler
    )

    print(f"--> Starting Optuna study '{study_name}' on {env_id} ({n_trials} trials)...")
    print(f"--> Persistent DB: {storage_path}")

    study.optimize(
        lambda t: objective(t, env_id=env_id, total_timesteps_per_trial=total_timesteps_per_trial, num_envs=num_envs),
        n_trials=n_trials,
        catch=(torch.cuda.OutOfMemoryError, Exception)
    )

    print("\n" + "=" * 60)
    print(f"OPTUNA STUDY COMPLETE FOR {env_id}")
    print("=" * 60)
    print(f"Best Trial #{study.best_trial.number}: Return = {study.best_value:.2f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for Atari Qwen")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timesteps-per-trial", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--storage", type=str, default="sqlite:///results/atari_qwen/optuna.db")
    parser.add_argument("--study-name", type=str, default="qwen_breakout_tuning")
    args = parser.parse_args()

    run_optuna_study(
        env_id=args.env_id,
        n_trials=args.n_trials,
        storage_path=args.storage,
        study_name=args.study_name,
        total_timesteps_per_trial=args.timesteps_per_trial,
        num_envs=args.num_envs
    )
