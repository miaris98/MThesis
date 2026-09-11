#!/usr/bin/env python3
"""Push the locally-pulled Phase 4 snapshot's bench2drive_out to Hugging Face.

Run this yourself (it reads HF_TOKEN from .env): a single push each time you
call it, or wrap it in your own loop (e.g. `while ($true) { py hf_push_phase4.py; sleep 600 }`
in PowerShell) if you want it repeating alongside sync_phase4_pull_loop.py.

Usage:
    py hf_push_phase4.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.config import paths  # noqa: E402

TAG = "phase4_live"
HF_REPO_ID = "Miaris/mthesis-wor-checkpoints"


def load_hf_token() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("HF_TOKEN not found in .env")


def main() -> int:
    b2d_dir = paths.imports_dir() / TAG / "bench2drive_out"
    if not b2d_dir.is_dir():
        raise SystemExit(
            "No bench2drive_out at %s yet - run sync_phase4_pull_loop.py "
            "(or sync_experiments.py --include results --tag %s) first." % (b2d_dir, TAG)
        )

    from huggingface_hub import HfApi
    api = HfApi(token=load_hf_token())
    api.upload_folder(
        repo_id=HF_REPO_ID,
        repo_type="model",
        folder_path=str(b2d_dir),
        path_in_repo="results/bench2drive_out",
        commit_message="Phase 4 live sync",
    )
    print("Pushed %s -> hf://%s/results/bench2drive_out" % (b2d_dir, HF_REPO_ID))
    return 0


if __name__ == "__main__":
    sys.exit(main())
