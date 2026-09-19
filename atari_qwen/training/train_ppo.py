"""High-throughput Vectorized PPO Training Loop with Qwen-Style Transformers for Atari."""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

from atari_qwen.config.atari_config import AtariConfig, get_config
from atari_qwen.envs.atari_wrappers import make_vector_atari_envs, make_atari_env
from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic


def parse_args() -> AtariConfig:
    parser = argparse.ArgumentParser(description="Train Qwen Transformer on Atari with PPO")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4", help="Atari environment ID")
    parser.add_argument("--preset", type=str, default="tiny", choices=["tiny", "small", "100m", "500m"], help="Model preset")
    parser.add_argument("--encoder-type", type=str, default="nature_cnn", choices=["nature_cnn", "impala_cnn", "patch"], help="Visual encoder")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128, help="Rollout steps per env")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total environment steps")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Peak learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Number of PPO minibatches")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs per rollout")
    parser.add_argument("--clip-coef", type=float, default=0.1, help="PPO surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Initial entropy coefficient")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp-name", type=str, default="qwen_atari_ppo", help="Experiment name")
    parser.add_argument("--log-dir", type=str, default="results/atari_qwen", help="Directory for logs and checkpoints")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision")
    parser.add_argument("--track-mlflow", action="store_true", help="Log metrics to MLflow")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None, help="Path to pretrained model checkpoint (.pt)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on policy model for maximum GPU speed")
    
    args = parser.parse_args()
    
    config = get_config(
        env_id=args.env_id,
        model_preset=args.preset,
        encoder_type=args.encoder_type,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        seed=args.seed,
        exp_name=args.exp_name,
        log_dir=args.log_dir,
        use_amp=not args.no_amp,
        track_mlflow=args.track_mlflow
    )
    config.pretrained_checkpoint = args.pretrained_checkpoint
    config.compile_model = args.compile
    return config


def evaluate_policy(
    model: QwenAtariActorCritic,
    env_id: str,
    device: torch.device,
    num_episodes: int = 10,
    seed: int = 123,
    max_eval_steps: int = 2500
) -> Tuple[float, float]:
    """Run deterministic policy evaluation across independent episodes."""
    eval_env_fn = make_atari_env(env_id, seed=seed, idx=0, noop_max=0, clip_reward=False, episodic_life=False)
    eval_env = eval_env_fn()
    action_meanings = eval_env.unwrapped.get_action_meanings()
    has_fire = "FIRE" in action_meanings and len(action_meanings) >= 2
    
    episode_returns = []
    model.eval()
    
    for ep in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_ret = 0.0
        step_count = 0
        stuck_counter = 0
        last_lives = eval_env.unwrapped.ale.lives()
        
        while not done and step_count < max_eval_steps:
            step_count += 1
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                actor_repr, _ = model._forward_transformer(obs_t)
                logits = model.actor_head(actor_repr)
                action = torch.argmax(logits, dim=-1).item()

            current_lives = eval_env.unwrapped.ale.lives()
            if has_fire and (current_lives < last_lives or (reward == 0 and stuck_counter > 50 if 'reward' in locals() else False)):
                action = 1
                stuck_counter = 0
            last_lives = current_lives
                
            step_result = eval_env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result
            ep_ret += reward
            if reward != 0:
                stuck_counter = 0
            else:
                stuck_counter += 1
            
        episode_returns.append(ep_ret)
        
    eval_env.close()
    model.train()
    return float(np.mean(episode_returns)), float(np.std(episode_returns))


def train(config: AtariConfig):
    # Setup directories
    run_name = f"{config.exp_name}_{config.env_id}_{config.model_preset}_{config.seed}_{int(time.time())}"
    run_dir = Path(config.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard setup
    writer = SummaryWriter(str(run_dir / "tb"))
    
    # MLflow setup (optional)
    mlflow_run = None
    if config.track_mlflow:
        try:
            import mlflow
            if config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment("atari_qwen")
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "env_id": config.env_id,
                "preset": config.model_preset,
                "encoder_type": config.encoder_type,
                "num_envs": config.num_envs,
                "learning_rate": config.learning_rate,
                "total_timesteps": config.total_timesteps,
            })
            print("✓ MLflow tracking enabled!")
        except Exception as e:
            print(f"--> Notice: MLflow tracking disabled ({e})")

    # Seeds & Device
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        device = torch.device("cuda")
        # Check bfloat16 support (e.g. Ampere / Ada Lovelace)
        if config.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        print(f"--> Using CUDA Device: {torch.cuda.get_device_name(0)} (AMP: {amp_dtype})")
    else:
        device = torch.device("cpu")
        amp_dtype = torch.float32
        config.use_amp = False
        print("--> Warning: CUDA unavailable, running on CPU.")

    # Create vectorized environments
    print(f"--> Initializing {config.num_envs} vectorized environments for {config.env_id}...")
    envs = make_vector_atari_envs(
        env_id=config.env_id,
        num_envs=config.num_envs,
        seed=config.seed,
        noop_max=config.noop_max,
        frame_stack=config.frame_stack,
        clip_reward=config.reward_clipping,
        episodic_life=config.episodic_life,
        asynchronous=True
    )
    action_dim = envs.single_action_space.n
    print(f"✓ Vectorized environments ready! Discrete Action Space: {action_dim}")

    # Build Qwen Policy Network
    model = QwenAtariActorCritic(
        action_dim=action_dim,
        in_channels=config.frame_stack,
        preset=config.model_preset,
        encoder_type=config.encoder_type,
        patch_size=config.patch_size,
        dropout=config.dropout,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        block_type=getattr(config, "block_type", "qwen"),
        impala_kaiming_init=getattr(config, "impala_kaiming_init", False),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Qwen Atari Policy initialized! Total Params: {param_count:,} ({param_count/1e6:.2f}M), Trainable: {trainable_count:,}")

    # Load Pretrained Weights if specified
    if getattr(config, "pretrained_checkpoint", None) and os.path.exists(config.pretrained_checkpoint):
        print(f"--> Loading pretrained weights from: {config.pretrained_checkpoint}")
        pre_ckpt = torch.load(config.pretrained_checkpoint, map_location=device, weights_only=False)
        sd = pre_ckpt.get("model_state_dict", pre_ckpt)
        model.load_state_dict(sd, strict=False)
        print("✓ Pretrained weights successfully loaded into Qwen policy network!")

    # Optional torch.compile for maximum GPU throughput
    raw_model = model
    if getattr(config, "compile_model", False):
        try:
            print("--> Compiling Qwen policy with torch.compile(backend='inductor')...")
            model = torch.compile(model)
            print("✓ torch.compile successfully enabled!")
        except Exception as e:
            print(f"--> Warning: torch.compile failed ({e}), continuing uncompiled.")
            model = raw_model

    # Optimizer & Scaler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, eps=1e-5, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=(config.use_amp and amp_dtype == torch.float16))

    # Rollout Storage Buffers
    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = config.total_timesteps // batch_size

    obs_buffer = torch.zeros((config.num_steps, config.num_envs, config.frame_stack, 84, 84), dtype=torch.uint8)
    actions_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long)
    logprobs_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32)
    rewards_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32)
    dones_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32)
    values_buffer = torch.zeros((config.num_steps, config.num_envs), dtype=torch.float32)

    # Initial Environment State
    global_step = 0
    start_time = time.time()
    reset_res = envs.reset()
    next_obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res
    next_done = torch.zeros(config.num_envs, device=device)
    next_obs = torch.as_tensor(next_obs, device=device)

    best_eval_return = -float("inf")
    print(f"--> Starting PPO training: {num_updates} updates ({config.total_timesteps:,} total steps)...")

    for update in range(1, num_updates + 1):
        # Anneal Learning Rate
        frac = 1.0 - (update - 1.0) / num_updates
        if config.lr_schedule == "cosine":
            lr_now = config.learning_rate * 0.5 * (1.0 + np.cos(np.pi * (1.0 - frac)))
        elif config.lr_schedule == "linear":
            lr_now = frac * config.learning_rate
        else:
            lr_now = config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        # Anneal Entropy Coefficient
        ent_coef_now = config.ent_coef_end + frac * (config.ent_coef - config.ent_coef_end)

        # ---------------- 1. Collect Rollout ----------------
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

            # Step environment
            step_result = envs.step(action.cpu().numpy())
            if len(step_result) == 5:
                next_obs_np, reward, terminated, truncated, infos = step_result
                done_np = np.logical_or(terminated, truncated)
            else:
                next_obs_np, reward, done_np, infos = step_result
                
            rewards_buffer[step] = torch.as_tensor(reward, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, device=device)
            next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            # Log episodic returns from wrapper infos if present
            if isinstance(infos, dict) and "episode" in infos:
                for ep_info in infos["episode"]:
                    if ep_info is not None:
                        ep_r = ep_info["r"]
                        ep_l = ep_info["l"]
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)

        # ---------------- 2. Generalized Advantage Estimation (GAE) ----------------
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

        # Flatten the rollout batch
        b_obs = obs_buffer.reshape((-1, config.frame_stack, 84, 84))
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        # ---------------- 3. PPO Optimization Epochs ----------------
        model.train()
        b_inds = np.arange(batch_size)
        clipfracs = []

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

                    # Policy loss
                    logratio = newlogprob - mb_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss with clipping
                    if config.clip_vloss:
                        v_loss_unclipped = (newvalue - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -config.clip_coef, config.clip_coef)
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()
                    
                    # Total loss
                    loss = pg_loss - ent_coef_now * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                if config.use_amp and amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

        # ---------------- 4. Telemetry & Checkpoints ----------------
        fps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", lr_now, global_step)
        writer.add_scalar("charts/entropy_coef", ent_coef_now, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/SPS", fps, global_step)

        if update % 10 == 0 or update == 1:
            print(f"Update {update:4d}/{num_updates:4d} | Step {global_step:8,d} | FPS: {fps:4d} | "
                  f"PLoss: {pg_loss.item():.4f} | VLoss: {v_loss.item():.4f} | Ent: {entropy_loss.item():.4f}")

        # Periodic Evaluation
        if global_step % config.eval_freq_steps < batch_size or update == num_updates:
            eval_mean, eval_std = evaluate_policy(
                model=model,
                env_id=config.env_id,
                device=device,
                num_episodes=config.eval_episodes,
                seed=config.seed + 1000
            )
            print(f"\n[EVALUATION] Step {global_step:,} | Mean Return: {eval_mean:.2f} +/- {eval_std:.2f}")
            writer.add_scalar("eval/mean_return", eval_mean, global_step)
            writer.add_scalar("eval/std_return", eval_std, global_step)

            # Checkpoint saving (save uncompiled weights for maximum portability)
            saved_model = getattr(model, "_orig_mod", getattr(model, "module", model))
            latest_path = ckpt_dir / "model_latest.pt"
            torch.save({
                "global_step": global_step,
                "model_state_dict": saved_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "eval_mean": eval_mean
            }, latest_path)

            if eval_mean > best_eval_return:
                best_eval_return = eval_mean
                best_path = ckpt_dir / "model_best.pt"
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": saved_model.state_dict(),
                    "eval_mean": eval_mean,
                    "config": config
                }, best_path)
                print(f"✓ Saved new best model: {best_path} (Return: {best_eval_return:.2f})\n")

    envs.close()
    writer.close()
    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metric("final_best_eval_return", best_eval_return)
            mlflow.end_run()
        except Exception:
            pass
    print("✓ Training completed successfully!")


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
