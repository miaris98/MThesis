"""Diagnostic: does the trained GTrXL+EZ2 actor's argmax action actually vary across
different observations, or has it collapsed to a single input-invariant constant action?

Motivation (struggle-solutions S-032/S-034): two unrelated training algorithms (an
off-policy replay-buffer trainer and an on-policy PPO+GAE trainer) both converged to the
exact same score (11.00) with the exact same eval action histogram (FIRE:45, LEFT:1280).
That coincidence is too specific to be architecture-independent learning -- it suggests
the actor's argmax may have collapsed to a constant action, with the eval harness's
scripted "FIRE on life loss / stuck" override the only source of variation, reproducibly
scoring the same fixed value against a fully deterministic emulator (fixed seed, no
sticky actions). This script feeds a battery of genuinely different observations through
a checkpoint and reports whether the argmax action (and the logits themselves) change.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.envs.atari_wrappers import make_atari_env


def load_agent(ckpt_path: str, action_dim: int, device: torch.device, use_gru_gating: bool = True) -> ImpalaGTrXLAgent:
    agent = ImpalaGTrXLAgent(
        action_dim=action_dim, in_channels=4, embed_dim=256, depth=4,
        num_heads=4, ffn_dim=1024, unroll_steps=5, bg_init=0.0,
        use_gru_gating=use_gru_gating,
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent


def diagnose(ckpt_path: str, env_id: str = "BreakoutNoFrameskip-v4", use_gru_gating: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_fn = make_atari_env(env_id, seed=999, idx=0, noop_max=0, clip_reward=False, episodic_life=False)
    env = env_fn()
    action_dim = env.action_space.n
    action_meanings = env.unwrapped.get_action_meanings()

    agent = load_agent(ckpt_path, action_dim, device, use_gru_gating=use_gru_gating)

    obs_batch = []
    labels = []

    # 1. Zero frame (blank screen)
    obs_batch.append(np.zeros((4, 84, 84), dtype=np.uint8))
    labels.append("all_zeros")

    # 2. Max-value frame (all white)
    obs_batch.append(np.full((4, 84, 84), 255, dtype=np.uint8))
    labels.append("all_255")

    # 3. Uniform random noise
    rng = np.random.RandomState(0)
    obs_batch.append(rng.randint(0, 256, size=(4, 84, 84), dtype=np.uint8))
    labels.append("random_noise_seed0")
    obs_batch.append(rng.randint(0, 256, size=(4, 84, 84), dtype=np.uint8))
    labels.append("random_noise_seed0_next")

    # 4. Real gameplay frames at several distinct points in an actual episode
    obs, _ = env.reset()
    obs_batch.append(obs.copy())
    labels.append("real_frame_t0_reset")

    real_action_seq = [1, 3, 3, 3, 0, 0, 3, 3, 0, 3, 3, 3, 0, 0, 3, 1, 3, 3, 0, 3]
    for i, a in enumerate(real_action_seq):
        step_res = env.step(a)
        if len(step_res) == 5:
            obs, r, term, trunc, _ = step_res
            done = term or trunc
        else:
            obs, r, done, _ = step_res
        if i in (4, 9, 14, 19):
            obs_batch.append(obs.copy())
            labels.append(f"real_frame_t{i+1}")
        if done:
            obs, _ = env.reset()

    env.close()

    print(f"--> Checkpoint: {ckpt_path}")
    print(f"--> Action meanings: {action_meanings}\n")
    print(f"{'Input':<28} {'Argmax action':<16} {'Logits'}")
    print("-" * 100)

    argmax_actions = []
    all_logits = []
    with torch.no_grad():
        for label, obs_np in zip(labels, obs_batch):
            obs_t = torch.as_tensor(obs_np, device=device).unsqueeze(0)
            logits, value, _ = agent(obs_t)
            probs = F.softmax(logits, dim=-1)
            action = torch.argmax(logits, dim=-1).item()
            argmax_actions.append(action)
            all_logits.append(logits.squeeze(0).cpu())
            action_name = action_meanings[action] if action < len(action_meanings) else str(action)
            logits_str = ", ".join(f"{x:.3f}" for x in logits.squeeze(0).cpu().tolist())
            probs_str = ", ".join(f"{x:.3f}" for x in probs.squeeze(0).cpu().tolist())
            print(f"{label:<28} {action_name:<16} logits=[{logits_str}] probs=[{probs_str}] value={value.item():.3f}")

    logits_stack = torch.stack(all_logits, dim=0)  # (num_inputs, action_dim)
    per_action_std = logits_stack.std(dim=0)
    print(f"\n--> Per-action logit std-dev across all {len(labels)} inputs: "
          f"[{', '.join(f'{x:.5f}' for x in per_action_std.tolist())}]")
    print(f"    (near-zero everywhere means the network is not reacting to input at all)")

    unique_actions = set(argmax_actions)
    print(f"\n--> Unique argmax actions across {len(argmax_actions)} wildly different inputs: {unique_actions}")
    if len(unique_actions) == 1:
        print("--> VERDICT: COLLAPSED -- argmax is a single constant action regardless of input.")
        print("    The network's output is input-invariant; any score variation across eval runs")
        print("    comes entirely from the eval harness's scripted FIRE-on-life-loss/stuck override")
        print("    acting on this constant action against a fully deterministic emulator.")
    else:
        print("--> VERDICT: NOT fully collapsed -- argmax varies across these inputs, so the")
        print("    identical 11.00 score across different training runs needs another explanation")
        print("    (e.g. a shared attractor basin in weight space, or coincidence).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--no-gru-gating", action="store_true",
                         help="Load the checkpoint as a plain-residual (no GRU gating) variant.")
    args = parser.parse_args()
    diagnose(args.ckpt, args.env_id, use_gru_gating=not args.no_gru_gating)
