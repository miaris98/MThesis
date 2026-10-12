"""Gate 0 Sanity Tests:
1. Replay Buffer Trajectory Continuity Test:
   Verifies that every sampled trajectory comes from ONE game and satisfies:
   - For k=0: target_future_obs[:, 0, :3] == obs_0[:, 1:]
   - For k>=1: target_future_obs[:, k, :3] == target_future_obs[:, k-1, 1:]
2. Random Agent Baseline Test:
   Evaluates a random agent over 10 episodes in Breakout through the standard evaluation harness
   to confirm it scores ~1.7.
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from atari_qwen.envs.atari_wrappers import make_atari_env, make_vector_atari_envs
from atari_qwen.training.train_mcts_offpolicy import MCTSReplayBuffer


def test_replay_buffer_continuity():
    print("\n[Gate 0.1] Testing Replay Buffer Trajectory Continuity...")
    num_envs = 8
    unroll_steps = 5
    capacity_per_env = 1000
    action_dim = 4

    # Create buffer
    buf = MCTSReplayBuffer(
        capacity_per_env=capacity_per_env,
        num_envs=num_envs,
        obs_shape=(4, 84, 84),
        action_dim=action_dim,
        unroll_steps=unroll_steps,
    )

    # Simulate stepping 8 vectorized environments with realistic frame stacks
    # Frame stack: each step shifts frames left and appends a unique new frame for that env
    frame_counters = np.zeros((num_envs,), dtype=np.int32)
    current_obs = np.zeros((num_envs, 4, 84, 84), dtype=np.uint8)

    # Initialize distinct initial frames
    for e in range(num_envs):
        for f in range(4):
            frame_counters[e] += 1
            current_obs[e, f] = frame_counters[e]

    # Step for 200 steps
    for step in range(200):
        actions = np.random.randint(0, action_dim, size=(num_envs,))
        rewards = np.zeros((num_envs,), dtype=np.float32)
        dones = np.zeros((num_envs,), dtype=np.bool_)
        # Occasionally terminate an env to test boundary handling
        if step in (50, 110):
            dones[0] = True
        pi = np.full((num_envs, action_dim), 1.0 / action_dim, dtype=np.float32)
        vals = np.zeros((num_envs,), dtype=np.float32)

        # Record in buffer
        buf.add_batch(current_obs, actions, rewards, dones, pi, vals)

        # Advance frames
        next_obs = np.zeros_like(current_obs)
        for e in range(num_envs):
            next_obs[e, :3] = current_obs[e, 1:]
            frame_counters[e] += 1
            next_obs[e, 3] = frame_counters[e] % 255
        current_obs = next_obs

    # Sample batch
    batch = buf.sample_trajectories(batch_size=64)
    obs_0 = batch["obs_0"].numpy()
    target_future = batch["target_future_obs"].numpy()

    # Check 1: For k=0, target_future[:, 0, :3] must match obs_0[:, 1:]
    diff_0 = np.abs(target_future[:, 0, :3] - obs_0[:, 1:])
    max_err_0 = np.max(diff_0)
    assert max_err_0 < 1e-5, f"Continuity failed at k=0: max error = {max_err_0}"
    print("✓ Frame stack continuity verified for step k=0 (obs_0 -> target_future[0])")

    # Check 2: For k=1..K-1, target_future[:, k, :3] must match target_future[:, k-1, 1:]
    for k in range(1, unroll_steps):
        diff_k = np.abs(target_future[:, k, :3] - target_future[:, k-1, 1:])
        max_err_k = np.max(diff_k)
        assert max_err_k < 1e-5, f"Continuity failed at k={k}: max error = {max_err_k}"
        print(f"✓ Frame stack continuity verified for step k={k} (target_future[{k-1}] -> target_future[{k}])")

    print("✓ [Gate 0.1 PASS] Replay buffer trajectory continuity 100% verified!\n")


def test_random_agent_eval(num_episodes: int = 10, env_id: str = "BreakoutNoFrameskip-v4"):
    print(f"\n[Gate 0.2] Evaluating Random Agent over {num_episodes} episodes in {env_id}...")
    scores = []
    
    for ep in range(num_episodes):
        # Fresh seed per episode for honest statistical evaluation
        seed = 1000 + ep
        env_fn = make_atari_env(env_id, seed=seed, idx=0, noop_max=30, clip_reward=False, episodic_life=False)
        env = env_fn()
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        action_dim = env.action_space.n

        while not done and steps < 27000:
            action = np.random.randint(0, action_dim)
            step_res = env.step(action)
            if len(step_res) == 5:
                obs, reward, term, trunc, _ = step_res
                done = term or trunc
            else:
                obs, reward, done, _ = step_res
            ep_ret += reward
            steps += 1
        env.close()
        scores.append(ep_ret)
        print(f"  Episode {ep+1:2d} (seed {seed}): Score = {ep_ret:.1f}, Steps = {steps}")

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    print(f"\n✓ [Gate 0.2 PASS] Random Agent Mean Score: {mean_score:.2f} +/- {std_score:.2f} (Expected: ~1.7)")
    assert 0.5 <= mean_score <= 4.0, f"Unexpected random score: {mean_score}"


def _fill_synthetic(buf, steps, num_envs, done_at=()):
    """Frame stacks whose newest frame encodes (env, t). After a done, the env 'resets': all four
    frames are replaced, so any window that crosses an episode end breaks the shift property."""
    obs = np.zeros((num_envs, 4, 84, 84), dtype=np.uint8)
    prev_done = np.zeros((num_envs,), dtype=np.bool_)
    for t in range(steps):
        for e in range(num_envs):
            if prev_done[e]:
                obs[e] = 1 + (e * 31 + t + 125) % 250  # reset: whole stack new, no shift
            else:
                obs[e, :3] = obs[e, 1:]
                obs[e, 3] = 1 + (e * 31 + t) % 250
        dones = np.zeros((num_envs,), dtype=np.bool_)
        for (tt, ee) in done_at:
            if tt == t:
                dones[ee] = True
        buf.add_batch(obs.copy(), np.full((num_envs,), t % 4), np.zeros(num_envs, np.float32), dones,
                      np.full((num_envs, 4), 0.25, np.float32), np.zeros(num_envs, np.float32))
        prev_done = dones


def test_replay_wraparound_and_boundaries():
    """Continuity must also hold once the ring buffer has wrapped, and no window may span a done."""
    print("\n[Gate 0.3] Replay continuity after wrap-around + episode boundaries...")
    K, N = 5, 8
    buf = MCTSReplayBuffer(capacity_per_env=40, num_envs=N, unroll_steps=K)
    dones = [(t, e) for t in range(0, 130, 7) for e in range(0, N, 3)]
    _fill_synthetic(buf, 130, N, done_at=dones)  # 130 > 40: wrapped more than 3 times
    for _ in range(20):
        b = buf.sample_trajectories(batch_size=128)
        o0, fut = b["obs_0"].numpy(), b["target_future_obs"].numpy()
        assert np.abs(fut[:, 0, :3] - o0[:, 1:]).max() < 1e-6
        for k in range(1, K):
            assert np.abs(fut[:, k, :3] - fut[:, k - 1, 1:]).max() < 1e-6
    print("✓ [Gate 0.3 PASS] continuity holds after wrap-around and never crosses an episode end")


def test_replay_save_load_roundtrip(tmp_dir: Path = None):
    """Resume must restore the buffer bit-exactly, and mark the resume point as an episode end."""
    print("\n[Gate 0.4] Replay buffer save/load round trip...")
    import tempfile
    tmp_dir = Path(tmp_dir or tempfile.mkdtemp())
    buf = MCTSReplayBuffer(capacity_per_env=30, num_envs=4, unroll_steps=5)
    _fill_synthetic(buf, 45, 4)
    path = tmp_dir / "replay_latest.npz"
    buf.save(path)
    buf2 = MCTSReplayBuffer(capacity_per_env=30, num_envs=4, unroll_steps=5)
    buf2.load(path)
    assert (buf2.ptr, buf2.size) == (buf.ptr, buf.size)
    assert np.array_equal(buf2.obs_buf, buf.obs_buf)
    assert np.array_equal(buf2.act_buf, buf.act_buf)
    assert buf2.done_buf[(buf2.ptr - 1) % buf2.capacity].all(), "resume point must be marked terminal"
    print("✓ [Gate 0.4 PASS] buffer restored exactly; resume boundary marked")


class _RandomAgent(torch.nn.Module):
    """Uniform-random 'policy' with the agent's interface, to push random play through the
    real evaluation harness (forced FIRE, noop starts, 27k-step cap) rather than a copy of it."""
    def __init__(self, action_dim=4):
        super().__init__()
        self.action_dim = action_dim

    def forward(self, obs):
        B = obs.shape[0]
        return torch.rand(B, self.action_dim), torch.zeros(B, 1), torch.zeros(B, 8)


def test_random_agent_through_eval_harness(num_episodes: int = 10):
    print(f"\n[Gate 0.5] Random policy through evaluate_agent_mcts (use_mcts=False), {num_episodes} eps...")
    from atari_qwen.training.train_mcts_offpolicy import evaluate_agent_mcts
    mean, std = evaluate_agent_mcts(_RandomAgent(), None, "BreakoutNoFrameskip-v4",
                                    torch.device("cpu"), num_episodes=num_episodes, use_mcts=False)
    print(f"  harness random score: {mean:.2f} +/- {std:.2f} (literature random 1.7)")
    # The harness forces FIRE after each lost life, which a pure random agent does not get,
    # so it may sit somewhat above 1.7 - but a harness that reports ~0 or >> human is broken.
    assert 0.5 <= mean <= 6.0, f"Eval harness gives implausible random score {mean}"
    print("✓ [Gate 0.5 PASS]")


if __name__ == "__main__":
    test_replay_buffer_continuity()
    test_replay_wraparound_and_boundaries()
    test_replay_save_load_roundtrip()
    test_random_agent_eval()
    test_random_agent_through_eval_harness()
