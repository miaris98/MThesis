"""Generic periodic background pull of an in-progress remote training run onto the
external disk, so a destroyed/dropped box costs at most one interval's progress
instead of the whole run.

This is the general-purpose replacement for writing a new one-off script (like
sync_phase4_pull_loop.py) per run: same tar-over-ssh pull() from sync_experiments.py,
parameterized instead of hard-coded, defaulting to the two groups that matter most for
surviving a box death - checkpoints and mlruns (see the checkpoint-sync-and-resource-usage
memory rule: every remote training launch should have one of these running for its
whole duration, started right after the training command, not pulled once at the end).

Usage:
    python scripts/sync/pull_loop.py --port 23445 --host root@1.2.3.4 --tag qwen30m_augmented
    python scripts/sync/pull_loop.py --port 23445 --host root@1.2.3.4 --tag my_run \\
        --include checkpoints mlruns tensorboard --interval 120

Cycles after the first successful one are incremental: only files modified since that cycle started (minus
--margin, for clock skew between the box and this machine and for writes in flight) are sent, and every
--full_every-th cycle is a full pull again. Every cycle used to re-send the whole tree, so a run with a few GB of
checkpoints moved terabytes a day at the 3-minute default.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.sync import sync_experiments as se  # noqa: E402
from src.config import paths  # noqa: E402


def pull_once(port: str, host: str, remote_root: str, includes: list, tag: str, since=None) -> bool:
    dest = paths.imports_dir() / tag
    try:
        se.pull(port, host, remote_root, includes, dest, since=since)
        return True
    except SystemExit as e:
        print("[pull] FAILED:", e, flush=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="SSH port")
    parser.add_argument("--host", required=True, help="user@host, e.g. root@1.2.3.4")
    parser.add_argument("--remote-root", default="/workspace")
    parser.add_argument("--tag", required=True, help="Destination subfolder under <EXP_ROOT>/imports/")
    parser.add_argument("--include", nargs="+", default=["checkpoints", "mlruns"],
                         metavar="GROUP", help="sync_experiments.py INCLUDE_GROUPS to pull each cycle")
    parser.add_argument("--interval", type=float, default=180.0, help="Seconds between pulls")
    parser.add_argument("--margin", type=float, default=900.0,
                        help="Seconds subtracted from the last successful cycle's start when choosing what to "
                             "re-send; covers clock skew and files written while that cycle ran")
    parser.add_argument("--full_every", type=int, default=20,
                        help="Every Nth cycle is a full pull (0 = only the first)")
    parser.add_argument("--merge-mlflow", action="store_true",
                         help="Also merge any pulled mlruns into the local MLflow store each cycle")
    args = parser.parse_args()

    print(f"Pulling {args.host}:{args.remote_root} ({', '.join(args.include)}) "
          f"every {args.interval:.0f}s -> {paths.imports_dir() / args.tag}", flush=True)
    since, cycle = None, 0
    while True:
        started = time.time()
        full = since is None or (args.full_every > 0 and cycle % args.full_every == 0)
        cycle += 1
        try:
            if pull_once(args.port, args.host, args.remote_root, args.include, args.tag,
                         since=None if full else since):
                since = started - args.margin
            if args.merge_mlflow and "mlruns" in args.include:
                src_store = paths.imports_dir() / args.tag / "MThesis" / "mlruns"
                if src_store.exists():
                    se.merge_store(src_store, paths.exp_root() / "mlruns", args.tag)
        except Exception as e:  # noqa: BLE001 - keep the loop alive across a bad cycle
            print("[cycle] error:", e, flush=True)
        time.sleep(max(0.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    sys.exit(main())
