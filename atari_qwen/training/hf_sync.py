"""Hugging Face Hub synchronization for Atari Qwen checkpoints and experiment metrics."""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


DEFAULT_HF_REPO_ID = "Miaris/mthesis-atari-qwen"


def load_hf_token() -> str:
    """Load Hugging Face token from environment or .env file."""
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    
    # Check .env in repository root
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
                
    raise ValueError("HF_TOKEN not found! Please set HF_TOKEN environment variable or add it to .env")


def generate_model_card(local_dir: Path, repo_id: str, game_id: str, preset: str) -> str:
    """Generate markdown model card for Hugging Face Hub."""
    return f"""---
tags:
- reinforcement-learning
- atari-2600
- qwen-transformer
- ppo
- pytorch
library_name: transformers
---

# Qwen Atari Reinforcement Learning Policy

This repository contains trained checkpoints and telemetry for **Qwen-style Transformer Policies** applied to **Atari 2600** games.

- **Game:** `{game_id}`
- **Preset:** `{preset}`
- **Architecture:** Qwen Transformer Blocks with Trainable Alpha Residual Gating (`RMSNorm`, `SwiGLU`, `QwenTransformerBlock`)
- **Algorithm:** Vectorized Proximal Policy Optimization (PPO) with GAE and Automatic Mixed Precision (`bfloat16`/`float16`).

## Files
- `model_best.pt`: Best checkpoint based on deterministic evaluation.
- `model_latest.pt`: Most recent training checkpoint.
- `tb/`: TensorBoard rollout & loss telemetry.

Synchronized from training instance at: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
"""


def push_to_hub(
    local_dir: str,
    repo_id: str = DEFAULT_HF_REPO_ID,
    path_in_repo: Optional[str] = None,
    game_id: str = "BreakoutNoFrameskip-v4",
    preset: str = "tiny"
) -> bool:
    """Push checkpoints directory and model card to Hugging Face Hub."""
    if not HF_AVAILABLE:
        print("--> Error: huggingface_hub is not installed. Run 'pip install huggingface_hub'")
        return False

    token = load_hf_token()
    api = HfApi(token=token)

    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"--> Notice: Directory {local_dir} does not exist yet. Skipping push.")
        return False

    if path_in_repo is None:
        path_in_repo = local_path.name

    # Ensure repository exists
    try:
        create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)
    except Exception as e:
        print(f"--> Notice on create_repo: {e}")

    # Write model card if not present
    readme_path = local_path / "README.md"
    if not readme_path.exists():
        readme_path.write_text(generate_model_card(local_path, repo_id, game_id, preset), encoding="utf-8")

    print(f"--> Uploading {local_dir} -> hf://{repo_id}/{path_in_repo}...")
    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(local_path),
            path_in_repo=path_in_repo,
            commit_message=f"Sync checkpoints for {game_id} ({preset})",
        )
        print(f"✓ Successfully synced to Hugging Face Hub: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"--> Error pushing to Hugging Face: {e}")
        return False


def run_sync_loop(local_dir: str, repo_id: str, interval_seconds: int = 300, game_id: str = "BreakoutNoFrameskip-v4", preset: str = "tiny"):
    """Periodically push checkpoints every interval_seconds (default 5 minutes)."""
    print(f"--> Starting Hugging Face sync loop every {interval_seconds}s for {local_dir}...")
    while True:
        try:
            push_to_hub(local_dir=local_dir, repo_id=repo_id, game_id=game_id, preset=preset)
        except Exception as e:
            print(f"--> Sync loop error: {e}")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Atari Qwen checkpoints to Hugging Face Hub")
    parser.add_argument("--local-dir", type=str, required=True, help="Path to checkpoints directory")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_HF_REPO_ID, help="Hugging Face Model Repo ID")
    parser.add_argument("--path-in-repo", type=str, default=None, help="Path inside repository")
    parser.add_argument("--loop-seconds", type=int, default=0, help="If > 0, run recurring sync loop every N seconds")
    parser.add_argument("--game-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--preset", type=str, default="tiny")
    args = parser.parse_args()

    if args.loop_seconds > 0:
        run_sync_loop(args.local_dir, args.repo_id, args.loop_seconds, args.game_id, args.preset)
    else:
        push_to_hub(args.local_dir, args.repo_id, args.path_in_repo, args.game_id, args.preset)
