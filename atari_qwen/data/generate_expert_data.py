"""Generate expert Atari demonstration dataset using CleanRL Pretrained SOTA DQN model."""
import argparse
import io
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

from atari_qwen.envs.atari_wrappers import make_vector_atari_envs


class CleanRLNatureDQN(nn.Module):
    """DeepMind Nature DQN architecture matching official CleanRL pretrained weights."""
    def __init__(self, action_dim: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.float() / 255.0)


def load_expert_dqn(device: torch.device) -> CleanRLNatureDQN:
    """Download official CleanRL SOTA Breakout DQN weights (Score: 412.0)."""
    url = "https://huggingface.co/cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1/resolve/main/q_network.pth"
    print(f"--> Downloading official CleanRL SOTA Breakout DQN weights from Hugging Face...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        state_dict = torch.load(io.BytesIO(resp.read()), map_location=device, weights_only=True)
    model = CleanRLNatureDQN().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ Loaded SOTA DQN Expert Model (Score: 412.0)!")
    return model


def generate_expert_dataset(
    env_id: str = "BreakoutNoFrameskip-v4",
    num_envs: int = 16,
    total_transitions: int = 100_000,
    output_path: str = "data/atari_expert/breakout_expert_100k.npz",
    epsilon: float = 0.05,
    device_str: str = "auto"
) -> str:
    """Generate high-throughput expert rollout dataset with vectorized environments."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    expert = load_expert_dqn(device)

    print(f"\n========================================================")
    print(f"--> Generating Expert Dataset: {total_transitions:,} transitions")
    print(f"--> Parallel Envs: {num_envs} | Epsilon Exploration: {epsilon}")
    print(f"--> Destination: {output_path}")
    print(f"========================================================\n")

    envs = make_vector_atari_envs(
        env_id=env_id,
        num_envs=num_envs,
        seed=42,
        noop_max=0,
        frame_stack=4,
        clip_reward=False,
        episodic_life=True
    )

    obs_buffer = np.zeros((total_transitions, 4, 84, 84), dtype=np.uint8)
    actions_buffer = np.zeros((total_transitions,), dtype=np.int64)
    rewards_buffer = np.zeros((total_transitions,), dtype=np.float32)
    dones_buffer = np.zeros((total_transitions,), dtype=bool)

    obs_res = envs.reset()
    obs = obs_res[0] if isinstance(obs_res, tuple) else obs_res

    collected = 0
    start_time = time.time()
    last_lives = np.full(num_envs, 5)
    stuck_counters = np.zeros(num_envs, dtype=int)

    while collected < total_transitions:
        obs_t = torch.as_tensor(obs, device=device)
        with torch.no_grad():
            q_values = expert(obs_t)
            greedy_actions = torch.argmax(q_values, dim=-1).cpu().numpy()

        actions = greedy_actions.copy()
        if epsilon > 0:
            for i in range(num_envs):
                if np.random.rand() < epsilon:
                    actions[i] = np.random.choice([0, 1, 2, 3])

        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        # Batch insert into memory
        batch_count = min(num_envs, total_transitions - collected)
        obs_buffer[collected:collected + batch_count] = obs[:batch_count]
        actions_buffer[collected:collected + batch_count] = actions[:batch_count]
        rewards_buffer[collected:collected + batch_count] = rewards[:batch_count]
        dones_buffer[collected:collected + batch_count] = dones[:batch_count]

        collected += batch_count
        obs = next_obs

        if collected % 10_000 == 0 or collected >= total_transitions:
            elapsed = time.time() - start_time
            rate = collected / elapsed
            print(f"--> Collected: {collected:6,d} / {total_transitions:,} transitions ({collected/total_transitions*100:5.1f}%) | Speed: {rate:5.0f} samples/sec")

    envs.close()
    elapsed = time.time() - start_time
    print(f"\n--> Saving compressed expert dataset to {output_path}...")
    np.savez_compressed(
        output_path,
        obs=obs_buffer,
        actions=actions_buffer,
        rewards=rewards_buffer,
        dones=dones_buffer
    )
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Successfully generated expert dataset: {output_path} ({file_size_mb:.2f} MB in {elapsed:.1f}s)!\n")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert Breakout dataset")
    parser.add_argument("--total-transitions", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--output-path", type=str, default="data/atari_expert/breakout_expert_100k.npz")
    args = parser.parse_args()

    generate_expert_dataset(
        total_transitions=args.total_transitions,
        num_envs=args.num_envs,
        output_path=args.output_path
    )
