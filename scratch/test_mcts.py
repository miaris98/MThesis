"""Unit test for MCTS Latent Engine and Off-Policy Replay Buffer."""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.mcts.mcts_engine import MCTSEngine
from atari_qwen.training.train_mcts_offpolicy import MCTSReplayBuffer


def test_mcts_latent_expansion():
    print("--> Testing MCTSEngine latent expansion...")
    device = torch.device("cpu")
    agent = ImpalaGTrXLAgent(
        action_dim=4,
        in_channels=4,
        embed_dim=64,
        depth=1,
        num_heads=2,
        ffn_dim=128,
        unroll_steps=3,
    ).to(device)

    engine = MCTSEngine(action_dim=4, num_simulations=10)
    root_latent = torch.randn(64, device=device)
    root_policy = torch.randn(64, device=device)

    probs, action, val = engine.search_single(
        root_latent, agent, device, root_policy_repr=root_policy, add_dirichlet=True, temperature=1.0
    )

    assert len(probs) == 4, f"Expected 4 action probs, got {len(probs)}"
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5), f"Probs should sum to 1.0, got {np.sum(probs)}"
    assert 0 <= action < 4, f"Invalid action {action}"
    print(f"✓ MCTS single search passed! Probs: {probs}, Action: {action}, Value: {val:.4f}")

    # Test batch search
    batch_latents = torch.randn(4, 64, device=device)
    batch_pol = torch.randn(4, 64, device=device)
    b_probs, b_acts, b_vals = engine.search_batch(
        batch_latents, agent, device, root_policy_reprs=batch_pol, add_dirichlet=False, temperature=0.0
    )
    assert b_probs.shape == (4, 4)
    assert b_acts.shape == (4,)
    assert b_vals.shape == (4,)
    print(f"✓ MCTS batch search passed! Batch actions: {b_acts}")


def test_replay_buffer():
    print("--> Testing MCTSReplayBuffer...")
    buf = MCTSReplayBuffer(capacity=100, obs_shape=(4, 84, 84), action_dim=4, unroll_steps=3)
    
    # Add dummy transitions
    obs_batch = np.zeros((10, 4, 84, 84), dtype=np.uint8)
    act_batch = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    rew_batch = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
    done_batch = np.array([False] * 10)
    pi_batch = np.full((10, 4), 0.25, dtype=np.float32)
    val_batch = np.zeros(10, dtype=np.float32)

    buf.add_batch(obs_batch, act_batch, rew_batch, done_batch, pi_batch, val_batch)
    assert buf.size == 10

    batch = buf.sample_trajectories(batch_size=4)
    assert batch["obs_0"].shape == (4, 4, 84, 84)
    assert batch["actions_seq"].shape == (4, 3)
    assert batch["rewards_seq"].shape == (4, 3)
    assert batch["target_future_obs"].shape == (4, 3, 4, 84, 84)
    assert batch["pi_0"].shape == (4, 4)
    print("✓ MCTSReplayBuffer sample_trajectories passed!")


if __name__ == "__main__":
    test_mcts_latent_expansion()
    test_replay_buffer()
    print("\n✓ All MCTS unit tests passed successfully!")
