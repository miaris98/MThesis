#!/usr/bin/env python3
"""Periodic background pull of the in-progress Phase 4 Bench2Drive run onto the
external disk. Pulls ONLY /workspace/bench2drive_out - deliberately not through
sync_experiments.py's 'results' include group, which also bundles the older
closed_loop/closed_loop_official/closed_loop_seed1/closed_loop_seed2 video
directories (hundreds of MB, unrelated to Phase 4) that would otherwise get
re-transferred in full every single cycle. Registers a narrow include group of
its own and reuses sync_experiments.py's tar-over-ssh pull() unchanged.

No credentials involved - this only needs SSH access to the vast.ai instance.
For pushing the same snapshot to Hugging Face, run hf_push_phase4.py separately.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import sync_experiments as se  # noqa: E402
from src.config import paths  # noqa: E402

se.INCLUDE_GROUPS["phase4"] = [
    "bench2drive_out/cnn_full51.json",
    "bench2drive_out/cnn_full51_perroute",
    "bench2drive_out/qwen30m_geom_full51.json",
    "bench2drive_out/qwen30m_geom_full51_perroute",
    "bench2drive_out/tcp_full51.json",
    "bench2drive_out/tcp_full51_perroute",
]

PORT = "24315"
USER_HOST = "root@ssh9.vast.ai"
REMOTE_ROOT = "/workspace"
TAG = "phase4_live"
INTERVAL_S = 180


def pull_once() -> None:
    dest = paths.imports_dir() / TAG
    try:
        se.pull(PORT, USER_HOST, REMOTE_ROOT, ["phase4"], dest)
    except SystemExit as e:
        print("[pull] FAILED:", e)


def main() -> int:
    print("Pulling %s/bench2drive_out every %ds -> %s" % (
        USER_HOST, INTERVAL_S, paths.imports_dir() / TAG))
    while True:
        started = time.time()
        try:
            pull_once()
        except Exception as e:  # noqa: BLE001 - keep the loop alive across a bad cycle
            print("[cycle] error:", e)
        time.sleep(max(0.0, INTERVAL_S - (time.time() - started)))


if __name__ == "__main__":
    sys.exit(main())
