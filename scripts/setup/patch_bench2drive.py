#!/usr/bin/env python3
"""Make Bench2Drive's evaluator kill only its own CARLA server after a crash, not every server on its GPU.

`leaderboard/leaderboard/leaderboard_evaluator.py` (carla_garage leaderboard_2 branch) ends a crashed run
with

    cmd2 = "ps -ef | grep '-graphicsadapter=" + str(args.gpu_rank) + "' | grep -v grep | awk '{print $2}' | xargs -r kill -9"

That kills every CARLA server started with the same -graphicsadapter, so with several eval lanes on one
GPU (A9 ran 5 lanes on box AA's single A10), one lane's crash route kills its neighbours' simulators. Their
evaluators then sit in 600 s load_world time-outs, which is what eval_watchdog.sh and the crash-route
auto-skip in queue_A9c.sh kept reacting to (S-094, S-095, S-099). The grep also matches
-graphicsadapter=10 when gpu_rank is 1.

The replacement kills the evaluator's own server (`self.server`, the Popen started in _setup_simulation)
and its descendants through psutil's parent links. A process-group kill is not enough, because
`su carlauser -c` in our CARLA_ROOT shim can start a new session.

Idempotent: a patched file is detected by its marker and left alone. Run by `launch_b2d20.sh setup`.

    python scripts/setup/patch_bench2drive.py /workspace/carla_garage
"""
import sys
from pathlib import Path

MARKER = "MThesis patch: kill only this evaluator's own CARLA server"
OLD = """        if crashed:
            cmd2 = "ps -ef | grep '-graphicsadapter="+ str(args.gpu_rank) + "' | grep -v grep | awk '{print $2}' | xargs -r kill -9"
            server = subprocess.Popen(cmd2, shell=True, preexec_fn=os.setsid)
            atexit.register(os.killpg, server.pid, signal.SIGKILL)
"""
NEW = f"""        if crashed:
            # {MARKER} (scripts/setup/patch_bench2drive.py).
            # Upstream killed every CARLA started with this -graphicsadapter, i.e. the other lanes on the GPU.
            try:
                import psutil
                server = psutil.Process(self.server.pid)
                for child in server.children(recursive=True):
                    child.kill()
                server.kill()
            except Exception as e:  # already gone, or psutil missing: the run script's reaper still cleans up
                print("could not kill own CARLA server:", e, flush=True)
"""


def patch(garage: Path) -> str:
    path = garage / "Bench2Drive" / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"
    if not path.is_file():
        raise SystemExit(f"not found: {path}")
    src = path.read_text()
    if MARKER in src:
        return f"already patched: {path}"
    if OLD not in src:
        raise SystemExit(f"upstream code changed, patch not applied - check {path} by hand (search 'graphicsadapter')")
    path.write_text(src.replace(OLD, NEW, 1))
    return f"patched: {path}"


if __name__ == "__main__":
    print(patch(Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/carla_garage")))
