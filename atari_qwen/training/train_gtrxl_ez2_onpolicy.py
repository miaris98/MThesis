"""On-Policy PPO+GAE Trainer for IMPALA-CNN + GTrXL + EfficientZero v2.

Fallback architecture per struggle-solutions S-029/S-032: the off-policy replay-buffer
design (train_gtrxl_ez2_turbo.py) stabilizes under GAE-within-K-step-windows but plateaus,
oscillating between a small set of fixed-point behaviors rather than climbing (never
exceeded a score of 11.00 across a full 60k-step run). This trainer keeps the exact same
IMPALA-CNN + GTrXL + EfficientZero v2 architecture and auxiliary losses, but collects a
temporally continuous on-policy rollout (mirroring train_ppo.py's proven CleanRL-style
design) and computes true multi-step GAE over the whole rollout, instead of GAE over
independently-sampled, temporally-discontinuous K-step buffer windows.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

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

from atari_qwen.envs.atari_wrappers import make_vector_atari_envs
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.eval.evaluate import compute_hns
from atari_qwen.training.train_gtrxl_ez2_turbo import evaluate_agent


def _ez2_aux_losses(
    agent: ImpalaGTrXLAgent,
    obs_buffer: torch.Tensor,
    actions_buffer: torch.Tensor,
    rewards_buffer: torch.Tensor,
    dones_buffer: torch.Tensor,
    t_idx: torch.Tensor,
    env_idx: torch.Tensor,
    latent_z: torch.Tensor,
    unroll_steps: int,
    num_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """EfficientZero v2 multi-step reward/value/consistency losses over the real K future
    steps that actually follow each (t, env) position in the temporally continuous rollout
    buffer -- replacing the replay buffer's independently-sampled K-step windows. Positions
    too close to the end of the rollout, or whose window crosses an episode boundary
    (done=True), are masked out rather than used, since there is no valid "next state" for
    the dynamics unroll to predict there.
    """
    num_envs = obs_buffer.shape[1]
    len_ok = (t_idx + unroll_steps) < num_steps
    safe_t = torch.where(len_ok, t_idx, torch.zeros_like(t_idx))

    k_range = torch.arange(unroll_steps)
    env_cols = env_idx.unsqueeze(1).expand(-1, unroll_steps)
    done_window = dones_buffer[safe_t.unsqueeze(1) + k_range.unsqueeze(0), env_cols]
    no_done = done_window.sum(dim=1) == 0
    mask = len_ok & no_done

    zero = torch.tensor(0.0, device=device)
    if not mask.any():
        return zero, zero, zero, 0.0

    keep_t = safe_t[mask]
    keep_env = env_idx[mask]
    env_cols_v = keep_env.unsqueeze(1).expand(-1, unroll_steps)

    future_t = keep_t.unsqueeze(1) + 1 + k_range.unsqueeze(0)      # t+1 .. t+K
    window_t = keep_t.unsqueeze(1) + k_range.unsqueeze(0)          # t .. t+K-1

    future_obs = obs_buffer[future_t, env_cols_v].to(device)                # (Bv, K, 4, 84, 84)
    future_actions = actions_buffer[window_t, env_cols_v].to(device)        # (Bv, K)
    future_rewards = rewards_buffer[window_t, env_cols_v].to(device)        # (Bv, K)

    Bv = future_obs.shape[0]
    latent_z_valid = latent_z[mask.to(device)]
    flat_future = future_obs.reshape(Bv * unroll_steps, *future_obs.shape[2:])

    with torch.no_grad():
        flat_target_z, _ = agent.encode_observation(flat_future)
        flat_target_proj = agent.predictor.projector(flat_target_z)
        target_projs = flat_target_proj.view(Bv, unroll_steps, -1)
        future_values = agent.critic_head(flat_target_z).squeeze(-1).view(Bv, unroll_steps)

    unroll = agent.unroll_branches(latent_z_valid, future_actions)

    reward_loss = 0.0
    unroll_value_loss = 0.0
    consistency_loss = 0.0
    for k in range(unroll_steps):
        reward_loss = reward_loss + F.smooth_l1_loss(unroll["rewards"][k], future_rewards[:, k])
        unroll_value_loss = unroll_value_loss + F.smooth_l1_loss(unroll["values"][k].squeeze(-1), future_values[:, k])
        consistency_loss = consistency_loss + agent.predictor.compute_consistency_loss(unroll["projections"][k], target_projs[:, k])

    coverage = float(mask.float().mean().item())
    return reward_loss / unroll_steps, unroll_value_loss / unroll_steps, consistency_loss / unroll_steps, coverage


def train_onpolicy_gtrxl_ez2(
    env_id: str = "BreakoutNoFrameskip-v4",
    total_steps: int = 1_000_000,
    num_envs: int = 16,
    num_steps: int = 128,
    unroll_steps: int = 5,
    learning_rate: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    clip_coef: float = 0.1,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    ez_value_loss_weight: float = 0.25,
    reward_loss_weight: float = 1.0,
    consistency_loss_weight: float = 0.5,
    max_grad_norm: float = 1.0,
    eval_interval_updates: int = 20,
    log_dir: str = "results/atari_gtrxl_ez2_onpolicy",
    device_str: str = "auto",
    seed: int = 42,
    # Isolation test per struggle-solutions S-036/S-037: False replaces GTrXL's GRU gating
    # with plain residual addition, to test whether the gating mechanism itself is why the
    # actor's output stays input-invariant, independent of raw training step budget.
    use_gru_gating: bool = True,
    legacy_actor_init: bool = False,
    # Per S-038/E6: the critic's gradient into the shared trunk is ~65x the actor's and its
    # target is near-constant under sparse reward, which flattens the representation the actor
    # depends on. True stops value-loss gradients from reaching the trunk at all.
    detach_critic: bool = False,
    # Linearly anneal the entropy bonus from ent_coef to ent_coef_end over training. With
    # near-zero advantages the entropy term is the only consistent gradient and drives the
    # policy to uniform-for-every-input (S-038, E1/E2).
    ent_coef_end: float = None,
    # E21: near-zero raw advantages divided by a tiny std can itself inject high-variance noise
    # into the policy gradient. False skips the (adv - mean) / std normalization step.
    normalize_advantages: bool = True,
    # E38: floor the std in (adv - mean) / std at this value instead of the raw 1e-8 epsilon.
    # With near-zero sparse-reward advantages, std -> 0 makes the epsilon-guarded division blow
    # up into high-variance noise; clamping the denominator caps how hard that can amplify.
    adv_norm_std_floor: float = 1e-8,
    # E35/E36: passed straight through to evaluate_agent (train_gtrxl_ez2_turbo.py).
    eval_temperature: float = 0.0,
    eval_sticky_action_p: float = 0.0,
    # E41/E47/E26: passed straight through to ImpalaGTrXLAgent's constructor.
    cnn_orthogonal_init: bool = False,
    single_layer_actor_head: bool = False,
    norm_policy_repr: bool = True,
    # S-043: DECISIVE FIX -- see ImpalaGTrXLAgent's cnn_kaiming_init docstring.
    cnn_kaiming_init: bool = False,
    # E31: EZ2 aux loss weights (reward_loss_weight, ez_value_loss_weight,
    # consistency_loss_weight) are scaled by 0.0 until this many env steps have elapsed, then
    # ramp to their full value over the following aux_warmup_steps -- lets the policy/value
    # functions establish basic signal before the auxiliary predictive losses (which every prior
    # run showed pulling the shared trunk toward representation collapse) turn on.
    aux_warmup_steps: int = 0,
    # E37: separate learning rate for actor_head params vs everything else (trunk + critic +
    # predictor), inverting the usual ratio so the actor isn't outpaced by the critic pulling on
    # the shared trunk. None means "use learning_rate for everything" (previous behavior).
    actor_lr: float = None,
    # E33: small per-alive-frame reward shaping bonus added to the clipped env reward, giving a
    # dense gradient signal before the sparse brick-break reward is ever hit. 0.0 disables it.
    living_reward: float = 0.0,
    # PPO-mechanics audit (S-040 follow-up): train_ppo.py (the S-029 run that genuinely climbed
    # 0.00->17.00 with QwenAtariActorCritic) differs from this trainer in ways nothing in
    # S-024-S-040 has isolated -- it anneals the LR (cosine by default) instead of holding it
    # fixed, clips the value loss (PPO2-style) instead of plain MSE, and uses weight_decay=1e-2
    # instead of 1e-4. None of these were suspects before because the investigation had been
    # focused on the representation/architecture side; testing them now that E37 shows the
    # representation itself can be fixed without unlocking eval-score improvement.
    lr_schedule: str = "constant",  # "constant" (previous behavior), "cosine", "linear"
    clip_vloss: bool = False,       # False reproduces previous (unclipped) behavior
    weight_decay: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else device_str)
    run_name = f"gtrxl_ez2_onpolicy_{env_id}_{int(time.time())}"
    run_dir = Path(log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("==================================================================")
    print("--> Initializing On-Policy IMPALA + GTrXL + EfficientZero v2 (PPO+GAE)")
    print(f"--> Target:        {total_steps:,} steps | Vector Envs: {num_envs} | Rollout: {num_steps}")
    print(f"--> Unroll Steps:  {unroll_steps} | Environment: {env_id} | Device: {device}")
    print("==================================================================\n")

    envs = make_vector_atari_envs(env_id, num_envs=num_envs, seed=seed, noop_max=30, clip_reward=True, episodic_life=True)
    action_dim = envs.single_action_space.n

    agent = ImpalaGTrXLAgent(
        action_dim=action_dim,
        in_channels=4,
        embed_dim=256,
        depth=4,
        num_heads=4,
        ffn_dim=1024,
        unroll_steps=unroll_steps,
        # See struggle-solutions S-035: bg_init=2.0 was still frozen at init after 15k-60k steps
        # in both trainers, making the actor's output input-invariant. 0.0 lets gates open sooner.
        bg_init=0.0,
        use_gru_gating=use_gru_gating,
        legacy_actor_init=legacy_actor_init,
        cnn_orthogonal_init=cnn_orthogonal_init,
        cnn_kaiming_init=cnn_kaiming_init,
        single_layer_actor_head=single_layer_actor_head,
        norm_policy_repr=norm_policy_repr,
    ).to(device)

    if ent_coef_end is None:
        ent_coef_end = ent_coef

    if actor_lr is not None:
        actor_param_ids = {id(p) for p in agent.actor_head.parameters()}
        actor_params = [p for p in agent.parameters() if id(p) in actor_param_ids]
        other_params = [p for p in agent.parameters() if id(p) not in actor_param_ids]
        optimizer = optim.AdamW(
            [
                {"params": actor_params, "lr": actor_lr, "base_lr": actor_lr},
                {"params": other_params, "lr": learning_rate, "base_lr": learning_rate},
            ],
            eps=1e-5, weight_decay=weight_decay,
        )
    else:
        optimizer = optim.AdamW(agent.parameters(), lr=learning_rate, eps=1e-5, weight_decay=weight_decay)
        for pg in optimizer.param_groups:
            pg["base_lr"] = learning_rate
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    batch_size = num_envs * num_steps
    minibatch_size = max(1, batch_size // num_minibatches)
    num_updates = max(1, total_steps // batch_size)

    obs_buffer = torch.zeros((num_steps, num_envs, 4, 84, 84), dtype=torch.uint8)
    actions_buffer = torch.zeros((num_steps, num_envs), dtype=torch.long)
    logprobs_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    rewards_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    dones_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32)
    values_buffer = torch.zeros((num_steps, num_envs), dtype=torch.float32)

    reset_res = envs.reset()
    next_obs_np = reset_res[0] if isinstance(reset_res, tuple) else reset_res
    next_obs = torch.as_tensor(next_obs_np, device=device)
    next_done = torch.zeros(num_envs, device=device)

    global_step = 0
    start_time = time.time()
    best_eval = -float("inf")

    print(f"--> Starting on-policy PPO+GAE+EZ2 training: {num_updates} updates ({total_steps:,} total steps)...")

    for update in range(1, num_updates + 1):
        # PPO-mechanics audit: anneal LR per param group, same schedule as train_ppo.py's proven
        # S-029 recipe. Every prior run in this investigation held LR fixed the whole time.
        lr_frac = 1.0 - (update - 1.0) / num_updates
        for pg in optimizer.param_groups:
            if lr_schedule == "cosine":
                pg["lr"] = pg["base_lr"] * 0.5 * (1.0 + np.cos(np.pi * (1.0 - lr_frac)))
            elif lr_schedule == "linear":
                pg["lr"] = lr_frac * pg["base_lr"]
            # "constant": leave pg["lr"] at its current (base_lr) value.

        # ---------------- 1. Collect a temporally continuous on-policy rollout ----------------
        agent.eval()
        for step in range(num_steps):
            global_step += num_envs
            obs_buffer[step] = next_obs.cpu()
            dones_buffer[step] = next_done.cpu()

            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits, value, _ = agent(next_obs)
                probs = F.softmax(logits.float(), dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
                logprob = dist.log_prob(action)

            values_buffer[step] = value.squeeze(-1).float().cpu()
            actions_buffer[step] = action.cpu()
            logprobs_buffer[step] = logprob.float().cpu()

            step_result = envs.step(action.cpu().numpy())
            if len(step_result) == 5:
                next_obs_np, reward, terminated, truncated, infos = step_result
                done_np = np.logical_or(terminated, truncated)
            else:
                next_obs_np, reward, done_np, infos = step_result

            step_reward = torch.as_tensor(reward, dtype=torch.float32)
            if living_reward != 0.0:
                # E33: dense per-alive-frame bonus so there's gradient signal before the sparse
                # brick-break reward is ever hit. Skipped on the terminal step of an episode so
                # it can't be farmed by just staying alive without playing.
                not_done = torch.as_tensor(1.0 - done_np.astype("float32"))
                step_reward = step_reward + living_reward * not_done
            rewards_buffer[step] = step_reward
            next_obs = torch.as_tensor(next_obs_np, device=device)
            next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            if isinstance(infos, dict) and "episode" in infos:
                for ep_info in infos["episode"]:
                    if ep_info is not None:
                        writer.add_scalar("charts/episodic_return", ep_info["r"], global_step)
                        writer.add_scalar("charts/episodic_length", ep_info["l"], global_step)

        # ---------------- 2. True multi-step GAE(lambda) over the full rollout ----------------
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            _, next_value, _ = agent(next_obs)
            next_value = next_value.squeeze(-1).float().cpu()

        advantages = torch.zeros_like(rewards_buffer)
        lastgaelam = torch.zeros(num_envs)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done.cpu()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buffer[t + 1]
                nextvalues = values_buffer[t + 1]
            delta = rewards_buffer[t] + gamma * nextvalues * nextnonterminal - values_buffer[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_buffer

        # Flatten in row-major order (flat_idx = t * num_envs + env) so t/env can be recovered
        # from any shuffled minibatch index for the EZ2 auxiliary-loss window lookups below.
        b_obs = obs_buffer.reshape(-1, 4, 84, 84)
        b_actions = actions_buffer.reshape(-1)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        # ---------------- 3. PPO-clip + EfficientZero v2 update epochs ----------------
        agent.train()
        b_inds = np.arange(batch_size)
        last_pg = last_v = last_ent = last_rew = last_cons = torch.tensor(0.0)
        last_coverage = 0.0
        last_logit_spread = 0.0

        frac = 1.0 - (update - 1.0) / num_updates
        ent_coef_now = ent_coef_end + frac * (ent_coef - ent_coef_end)
        raw_adv_absmean = advantages.abs().mean().item()
        reward_rate = rewards_buffer.abs().mean().item()

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = torch.as_tensor(b_inds[start:end], dtype=torch.long)
                t_idx = mb_inds // num_envs
                env_idx = mb_inds % num_envs

                mb_obs = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)
                mb_logprobs = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)

                if normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std().clamp(min=adv_norm_std_floor)

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits, value, latent_z = agent(mb_obs)
                    # Direct in-training measure of input-dependence: how much the logits vary
                    # ACROSS the different observations in this minibatch. Near-zero means the
                    # actor is producing the same output for every state (S-038).
                    last_logit_spread = logits.float().std(dim=0).mean().item()
                    log_probs_all = F.log_softmax(logits, dim=-1)
                    new_logprob = log_probs_all.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                    probs_all = F.softmax(logits, dim=-1)
                    entropy = -(probs_all * log_probs_all).sum(dim=-1)

                    logratio = (new_logprob - mb_logprobs).clamp(-20.0, 20.0)
                    ratio = logratio.exp()

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if detach_critic:
                        newvalue = agent.critic_head(latent_z.detach()).squeeze(-1)
                    else:
                        newvalue = value.squeeze(-1)
                    if clip_vloss:
                        v_loss_unclipped = (newvalue - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -clip_coef, clip_coef)
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                    entropy_loss = entropy.mean()

                    reward_loss, unroll_value_loss, consistency_loss, coverage = _ez2_aux_losses(
                        agent, obs_buffer, actions_buffer, rewards_buffer, dones_buffer,
                        t_idx, env_idx, latent_z, unroll_steps, num_steps, device,
                    )

                    # E31: ramp the EZ2 aux loss weights linearly from 0 at step 0 to their full
                    # value at aux_warmup_steps, instead of applying them at full strength from
                    # step 1. Every prior run enabling these losses showed them pulling the
                    # shared trunk toward representation collapse (S-036/S-038 E11); the idea is
                    # to let the policy/value functions establish real signal first.
                    aux_warmup_frac = 1.0 if aux_warmup_steps <= 0 else min(1.0, global_step / aux_warmup_steps)

                    total_loss = (
                        pg_loss
                        - ent_coef_now * entropy_loss
                        + vf_coef * v_loss
                        + aux_warmup_frac * reward_loss_weight * reward_loss
                        + aux_warmup_frac * ez_value_loss_weight * unroll_value_loss
                        + aux_warmup_frac * consistency_loss_weight * consistency_loss
                    )

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

                last_pg, last_v, last_ent = pg_loss, v_loss, entropy_loss
                last_rew, last_cons, last_coverage = reward_loss, consistency_loss, coverage

        sps = int(global_step / (time.time() - start_time))
        if update % 5 == 0 or update == 1:
            print(f"Update {update:5d}/{num_updates:5d} | Step {global_step:8,d} | SPS: {sps:4d} | "
                  f"PLoss: {last_pg.item():.4f} | VLoss: {last_v.item():.4f} | Ent: {last_ent.item():.4f} | "
                  f"RewLoss: {last_rew.item():.4f} | SimLoss: {last_cons.item():.4f} | "
                  f"LogitSpread: {last_logit_spread:.5f} | RawAdv: {raw_adv_absmean:.5f} | "
                  f"RewRate: {reward_rate:.5f} | EntCoef: {ent_coef_now:.4f}", flush=True)
        writer.add_scalar("losses/policy", last_pg.item(), global_step)
        writer.add_scalar("losses/value", last_v.item(), global_step)
        writer.add_scalar("losses/entropy", last_ent.item(), global_step)
        writer.add_scalar("losses/reward", last_rew.item(), global_step)
        writer.add_scalar("losses/consistency", last_cons.item(), global_step)
        writer.add_scalar("diag/logit_spread", last_logit_spread, global_step)
        writer.add_scalar("diag/raw_advantage_absmean", raw_adv_absmean, global_step)
        writer.add_scalar("diag/reward_rate", reward_rate, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        if update % eval_interval_updates == 0 or update == num_updates:
            mean_eval, std_eval = evaluate_agent(
                agent, env_id, device, num_episodes=5,
                eval_temperature=eval_temperature, sticky_action_p=eval_sticky_action_p,
            )
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
    print(f"✓ On-Policy GTrXL + EfficientZero v2 Complete! Peak Eval: {best_eval:.2f}")
    print("==================================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="On-Policy PPO+GAE Trainer for IMPALA + GTrXL + EZ2")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128)
    args = parser.parse_args()
    train_onpolicy_gtrxl_ez2(
        env_id=args.env_id,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
    )
