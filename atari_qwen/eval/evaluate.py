"""Evaluation harness and Human-Normalized Score (HNS) benchmark comparison."""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atari_qwen.envs.atari_wrappers import make_atari_env
from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic

# Official DeepMind / Bellemare et al. / Hessel et al. (Rainbow) Atari Baselines
ATARI_BASELINES: Dict[str, Dict[str, float]] = {
    "BreakoutNoFrameskip-v4": {
        "random": 1.7,
        "human": 30.5,
        "dqn": 401.2,
        "rainbow": 417.5,
        "ppo": 274.8,
    },
    "PongNoFrameskip-v4": {
        "random": -20.7,
        "human": 14.6,
        "dqn": 18.9,
        "rainbow": 20.9,
        "ppo": 20.7,
    },
    "SpaceInvadersNoFrameskip-v4": {
        "random": 148.0,
        "human": 1668.7,
        "dqn": 1976.0,
        "rainbow": 17510.0,
        "ppo": 1050.0,
    },
    "QbertNoFrameskip-v4": {
        "random": 163.9,
        "human": 13455.0,
        "dqn": 10596.0,
        "rainbow": 33817.5,
        "ppo": 14275.0,
    },
    "BeamRiderNoFrameskip-v4": {
        "random": 363.9,
        "human": 16926.5,
        "dqn": 6846.0,
        "rainbow": 16850.2,
        "ppo": 2390.0,
    },
    "SeaquestNoFrameskip-v4": {
        "random": 68.4,
        "human": 42054.7,
        "dqn": 5286.0,
        "rainbow": 15898.9,
        "ppo": 1205.0,
    }
}


def compute_hns(score: float, env_id: str) -> float:
    """Calculate Human-Normalized Score (HNS) in percent."""
    if env_id not in ATARI_BASELINES:
        return 0.0
    random_score = ATARI_BASELINES[env_id]["random"]
    human_score = ATARI_BASELINES[env_id]["human"]
    return ((score - random_score) / (human_score - random_score)) * 100.0


def evaluate_checkpoint(
    checkpoint_path: str,
    env_id: Optional[str] = None,
    num_episodes: int = 30,
    device_str: str = "auto",
    deterministic: bool = True
) -> Dict[str, Any]:
    """Load a trained checkpoint and evaluate over multiple deterministic episodes."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", None)

    if env_id is None and config is not None:
        env_id = config.env_id
    elif env_id is None:
        env_id = "BreakoutNoFrameskip-v4"

    print(f"--> Evaluating checkpoint: {checkpoint_path}")
    print(f"--> Environment: {env_id} | Episodes: {num_episodes} | Device: {device}")

    # Build evaluation environment (unclipped rewards, no episodic life for true score)
    eval_env_fn = make_atari_env(
        env_id=env_id,
        seed=1000,
        idx=0,
        noop_max=30,
        frame_stack=getattr(config, "frame_stack", 4),
        clip_reward=False,
        episodic_life=False
    )
    env = eval_env_fn()
    action_dim = env.action_space.n

    # Reconstruct model
    preset = getattr(config, "model_preset", "tiny") if config else "tiny"
    encoder_type = getattr(config, "encoder_type", "nature_cnn") if config else "nature_cnn"

    model = QwenAtariActorCritic(
        action_dim=action_dim,
        in_channels=getattr(config, "frame_stack", 4),
        preset=preset,
        encoder_type=encoder_type
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    episode_scores = []
    episode_lengths = []

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        score = 0.0
        length = 0

        while not done:
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                actor_repr, _ = model._forward_transformer(obs_t)
                logits = model.actor_head(actor_repr)
                if deterministic:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    probs = torch.distributions.Categorical(logits=logits)
                    action = probs.sample().item()

            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result

            score += reward
            length += 1

        episode_scores.append(score)
        episode_lengths.append(length)
        print(f"Episode {ep:2d}/{num_episodes:2d}: Score = {score:6.1f} | Length = {length:5d}")

    env.close()

    mean_score = float(np.mean(episode_scores))
    std_score = float(np.std(episode_scores))
    median_score = float(np.median(episode_scores))
    hns = compute_hns(mean_score, env_id)

    print("\n" + "=" * 60)
    print(f"RESULTS FOR {env_id}")
    print("=" * 60)
    print(f"Mean Score:            {mean_score:.2f} +/- {std_score:.2f}")
    print(f"Median Score:          {median_score:.2f}")
    if env_id in ATARI_BASELINES:
        print(f"Human-Normalized Score: {hns:.2f}%")
        print(f"  Random Baseline:     {ATARI_BASELINES[env_id]['random']}")
        print(f"  Human Baseline:      {ATARI_BASELINES[env_id]['human']}")
        print(f"  Nature DQN:          {ATARI_BASELINES[env_id]['dqn']}")
        print(f"  Rainbow SOTA:        {ATARI_BASELINES[env_id]['rainbow']}")
    print("=" * 60 + "\n")

    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "median_score": median_score,
        "hns": hns,
        "episode_scores": episode_scores
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Qwen Atari checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--env-id", type=str, default=None, help="Atari environment ID (defaults to config)")
    parser.add_argument("--num-episodes", type=int, default=30, help="Number of evaluation episodes")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        deterministic=not args.stochastic
    )
