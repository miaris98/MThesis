import subprocess
import time
from remote_config import load_config

cfg = load_config()


def main():
    print("--> 1. Terminating old un-resumed Atari processes (preserving CARLA)...")
    cfg.run_ssh(
        "kill -9 $(pgrep -f 'train_mcts_offpolicy') 2>/dev/null || true; "
        "kill -9 $(pgrep -f 'run_100k_benchmark') 2>/dev/null || true",
        check=False,
    )
    time.sleep(2)

    print("--> 2. Archiving non-38k Atari test directories...")
    cleanup_cmd = f"""
mkdir -p /workspace/archive_old_atari
for d in {cfg.remote_path("results/100k_benchmark/S049*")}; do
    if [ -d "$d" ]; then
        vname=$(basename "$d")
        mkdir -p "/workspace/archive_old_atari/$vname"
        for sub in "$d"/mcts_offpolicy_*; do
            if [ -d "$sub" ] && [[ "$sub" != *"1790086674"* ]]; then
                echo "Archiving $sub"
                mv "$sub" "/workspace/archive_old_atari/$vname/" 2>/dev/null || true
            fi
        done
    fi
done
"""
    res = cfg.run_ssh(cleanup_cmd, check=False)
    print(res.stdout)

    print("--> 3. Checking remaining 38k checkpoints in benchmark tree...")
    check_cmd = (
        f"cd {cfg.workspace} && "
        f"{cfg.python()} -c \""
        "import glob; "
        "for v in ['S049a_mcts_sim20_s42', 'S049b_mcts_sim35_s42', 'S049c_mcts_sim50_s42']: "
        f"  files = glob.glob(f'{cfg.remote_path(\"results/100k_benchmark\")}" + "/{v}/**/model_latest.pt', recursive=True); "
        "  print(f'{v}: {files}')"
        "\""
    )
    res = cfg.run_ssh(check_cmd, check=False)
    print(res.stdout)

    print("--> 4. Launching Atari 100k Benchmark (S049a, S049b, S049c) with --resume --start-step 38000 on cores 16-31...")
    launch_cmd = (
        f"cd {cfg.workspace} && "
        f"taskset -c 16-31 nohup {cfg.python()} atari_qwen/scripts/run_100k_benchmark.py "
        "--variants S049a_mcts_sim20 S049b_mcts_sim35 S049c_mcts_sim50 "
        "--seeds 42 --parallel --resume --start-step 38000 "
        "> /workspace/benchmark_mcts_resumed_38k.log 2>&1 &"
    )
    cfg.run_ssh(launch_cmd, check=False)
    print("Launch command dispatched.")
    time.sleep(5)

    print("\n--> 5. Checking launcher log and active processes...")
    status_cmd = """
echo '=== LAUNCHER LOG ==='
cat /workspace/benchmark_mcts_resumed_38k.log
echo ''
echo '=== ACTIVE ATARI PROCESSES ==='
ps aux | grep -E 'python.*(train_mcts|run_100k)' | grep -v grep || echo 'NO ATARI PROCESSES'
echo ''
echo '=== CARLA PROCESS (SHOULD BE RUNNING) ==='
ps aux | grep -E 'python.*train_wor' | grep -v grep || echo 'ALERT: CARLA NOT FOUND'
"""
    res = cfg.run_ssh(status_cmd, check=False)
    print(res.stdout)


if __name__ == "__main__":
    main()
