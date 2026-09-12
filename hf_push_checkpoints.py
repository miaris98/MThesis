#!/usr/bin/env python3
"""Periodically push a training checkpoint directory to Hugging Face Hub.

Exists for the same reason as `hf_push_phase4.py` and the fix in challenges_02.md
2.9: direct scp/tar-over-ssh from a rented instance back to this machine measured
15-25 KB/s (hours for a few hundred MB), while both ends are well-peered with HF's
edge independently. For a long unattended `train_wor.py` run this is the safety net
against losing progress if the instance has to be stopped or destroyed before the
run finishes and before a manual sync - `train_wor.py --save_freq` controls how much
work is ever at risk between pushes.

Run this ON THE TRAINING BOX (not locally - the checkpoints live there), pointed at
the same --save_dir passed to train_wor.py. One push:

    python hf_push_checkpoints.py --local_dir /workspace/checkpoints/wor_full8town/qwen30m_geom_s0 \
        --path_in_repo checkpoints/wor_full8town/qwen30m_geom_s0

Or loop alongside training:

    nohup python hf_push_checkpoints.py --local_dir ... --path_in_repo ... --loop_seconds 900 &

HF_TOKEN is read from $HF_TOKEN if set, else from a .env file next to this script
(same convention as hf_push_phase4.py) - on a fresh box, copy .env up or export
HF_TOKEN before running.
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_HF_REPO_ID = "Miaris/mthesis-wor-checkpoints"


def load_hf_token() -> str:
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("HF_TOKEN not set: export it or put HF_TOKEN=... in a .env next to this script")


def push_once(api, local_dir: str, repo_id: str, path_in_repo: str) -> None:
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        commit_message="Checkpoint sync",
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pushed {local_dir} -> hf://{repo_id}/{path_in_repo}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local_dir", required=True, help="Directory to upload, e.g. --save_dir passed to train_wor.py")
    p.add_argument("--repo_id", default=DEFAULT_HF_REPO_ID, help="HF model repo to push into")
    p.add_argument("--path_in_repo", required=True, help="Destination path inside the repo")
    p.add_argument("--loop_seconds", type=float, default=0.0,
                   help="If >0, push repeatedly on this interval instead of once (Ctrl+C or kill to stop)")
    args = p.parse_args()

    if not Path(args.local_dir).is_dir():
        raise SystemExit(f"--local_dir does not exist yet: {args.local_dir}")

    from huggingface_hub import HfApi
    api = HfApi(token=load_hf_token())

    if args.loop_seconds <= 0:
        push_once(api, args.local_dir, args.repo_id, args.path_in_repo)
        return 0

    print(f"Looping every {args.loop_seconds:.0f}s. Ctrl+C to stop.")
    while True:
        try:
            push_once(api, args.local_dir, args.repo_id, args.path_in_repo)
        except Exception as e:
            print(f"[Warning] push failed, will retry next interval: {e}")
        time.sleep(args.loop_seconds)


if __name__ == "__main__":
    sys.exit(main())
