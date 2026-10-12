"""E1/E2: find the exact layer where across-input variation collapses.

Feeds a battery of maximally-different observations through the network and reports, at every
stage of the forward pass, how much the activations actually differ across those inputs. Run it
on a trained checkpoint (E1) and on a fresh random init (E2) -- if the random init shows healthy
variance and the trained one does not, training is destroying the signal rather than the
architecture being born broken.

Metric per stage:
  abs_std  = mean over features of (std across the N inputs)
  rel_std  = abs_std / (mean over features of |activation|)   <- scale-free, the one that matters
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.envs.atari_wrappers import make_atari_env


def build_inputs(env_id: str, device: torch.device):
    env_fn = make_atari_env(env_id, seed=999, idx=0, noop_max=0, clip_reward=False, episodic_life=False)
    env = env_fn()
    obs_list, labels = [], []

    obs_list.append(np.zeros((4, 84, 84), dtype=np.uint8)); labels.append("all_zeros")
    obs_list.append(np.full((4, 84, 84), 255, dtype=np.uint8)); labels.append("all_255")
    rng = np.random.RandomState(0)
    obs_list.append(rng.randint(0, 256, size=(4, 84, 84), dtype=np.uint8)); labels.append("noise_a")
    obs_list.append(rng.randint(0, 256, size=(4, 84, 84), dtype=np.uint8)); labels.append("noise_b")

    obs, _ = env.reset()
    obs_list.append(obs.copy()); labels.append("real_t0")
    for i, a in enumerate([1, 3, 3, 3, 0, 0, 3, 3, 0, 3, 3, 3, 0, 0, 3, 1, 3, 3, 0, 3]):
        step_res = env.step(a)
        obs, done = (step_res[0], step_res[2] or step_res[3]) if len(step_res) == 5 else (step_res[0], step_res[2])
        if i in (4, 9, 14, 19):
            obs_list.append(obs.copy()); labels.append(f"real_t{i+1}")
        if done:
            obs, _ = env.reset()
    env.close()

    batch = torch.as_tensor(np.stack(obs_list), device=device)  # (N, 4, 84, 84)
    return batch, labels


def report(name: str, acts: torch.Tensor, rows: list):
    """acts: (N, ...) -- activations for N different inputs."""
    flat = acts.reshape(acts.shape[0], -1).float()
    abs_std = flat.std(dim=0).mean().item()
    scale = flat.abs().mean().item()
    rel_std = abs_std / (scale + 1e-12)
    rows.append((name, abs_std, scale, rel_std))
    print(f"{name:<34} abs_std={abs_std:.6f}  scale={scale:.6f}  rel_std={rel_std:.6f}")


def probe(ckpt_path: str, env_id: str, use_gru_gating: bool, random_init: bool,
          single_layer_actor_head: bool = False, norm_policy_repr: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, labels = build_inputs(env_id, device)
    action_dim = 4

    agent = ImpalaGTrXLAgent(
        action_dim=action_dim, in_channels=4, embed_dim=256, depth=4,
        num_heads=4, ffn_dim=1024, unroll_steps=5, bg_init=0.0,
        use_gru_gating=use_gru_gating,
        # Must match the checkpoint's architecture (E47/E26) or load_state_dict fails/silently
        # mismatches shapes.
        single_layer_actor_head=single_layer_actor_head,
        norm_policy_repr=norm_policy_repr,
    ).to(device)

    if random_init:
        print("--> RANDOM INIT (untrained network)")
    else:
        agent.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        print(f"--> Trained checkpoint: {ckpt_path}")
    agent.eval()

    print(f"--> {batch.shape[0]} inputs: {labels}")
    print(f"--> use_gru_gating={use_gru_gating}\n")
    print(f"{'Stage':<34} {'across-input spread'}")
    print("-" * 92)

    rows = []
    enc = agent.visual_encoder
    with torch.no_grad():
        x = batch.float()
        report("00_raw_obs", x, rows)
        if x.max() > 1.0:
            x = x / 255.0
        report("01_normalized_obs", x, rows)

        x = enc.stage1(x); report("02_cnn_stage1", x, rows)
        x = enc.stage2(x); report("03_cnn_stage2", x, rows)
        x = enc.stage3(x); report("04_cnn_stage3", x, rows)

        B, C, H, W = x.shape
        tok = x.flatten(2).transpose(1, 2)
        tok = enc.proj(tok); report("05_cnn_proj", tok, rows)
        vis_tokens = enc.layer_norm(tok); report("06_cnn_layernorm(vis_tokens)", vis_tokens, rows)

        N = batch.shape[0]
        state_tok = agent.state_token.expand(N, -1, -1)
        pol_tok = agent.policy_token.expand(N, -1, -1)
        seq = torch.cat([state_tok, pol_tok, vis_tokens], dim=1)
        seq = seq + agent.pos_embed[:, : seq.shape[1], :]
        report("07_seq_with_query_tokens", seq, rows)
        report("07b_policy_token_slot_only", seq[:, 1], rows)

        for i, blk in enumerate(agent.blocks):
            seq = blk(seq)
            report(f"08_after_block{i}", seq, rows)
            report(f"08b_after_block{i}_policy_slot", seq[:, 1], rows)

        seq = agent.norm(seq)
        latent_z = seq[:, 0]
        policy_repr = seq[:, 1]
        report("09_latent_z", latent_z, rows)
        report("10_policy_repr", policy_repr, rows)

        hidden = agent.actor_head[0](policy_repr)
        report("11_actor_head_hidden", hidden, rows)
        logits = agent.actor_head(policy_repr)
        report("12_logits", logits, rows)
        value = agent.critic_head(latent_z)
        report("13_value", value, rows)

    print("\n--> Where the signal dies (first big rel_std drop):")
    prev = None
    for name, abs_std, scale, rel in rows:
        if prev is not None and prev[3] > 1e-6:
            ratio = rel / (prev[3] + 1e-12)
            if ratio < 0.2:
                print(f"    {prev[0]} -> {name}: rel_std fell {prev[3]:.6f} -> {rel:.6f} ({ratio:.3f}x)")
        prev = (name, abs_std, scale, rel)

    print(f"\n--> Actor head weight scales:")
    for i, m in enumerate(agent.actor_head):
        if isinstance(m, torch.nn.Linear):
            print(f"    actor_head[{i}] weight abs-mean={m.weight.abs().mean().item():.6f} "
                  f"std={m.weight.std().item():.6f} shape={tuple(m.weight.shape)}")
    for i, m in enumerate(agent.critic_head):
        if isinstance(m, torch.nn.Linear):
            print(f"    critic_head[{i}] weight abs-mean={m.weight.abs().mean().item():.6f} "
                  f"std={m.weight.std().item():.6f} shape={tuple(m.weight.shape)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--no-gru-gating", action="store_true")
    parser.add_argument("--random-init", action="store_true",
                        help="Probe a fresh untrained network instead of a checkpoint (E2).")
    parser.add_argument("--single-layer-actor-head", action="store_true",
                        help="Match a checkpoint trained with single_layer_actor_head=True (E47).")
    parser.add_argument("--no-norm-policy-repr", action="store_true",
                        help="Match a checkpoint trained with norm_policy_repr=False (E26).")
    args = parser.parse_args()
    if not args.random_init and not args.ckpt:
        parser.error("--ckpt is required unless --random-init is given")
    probe(args.ckpt, args.env_id, use_gru_gating=not args.no_gru_gating, random_init=args.random_init,
          single_layer_actor_head=args.single_layer_actor_head,
          norm_policy_repr=not args.no_norm_policy_repr)
