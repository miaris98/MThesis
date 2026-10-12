"""Orchestrator for Gate 2: 20k steps.

Runs 3 parallel variants:
1. S050a_gate2_sim16  (cores from cfg.cpu_cores_a): 16 collection simulations, 16 eval simulations
2. S050b_gate2_sim32  (cores from cfg.cpu_cores_b): 32 collection simulations, 32 eval simulations
3. S050c_gate2_policy_only (cores from cfg.cpu_cores_c): policy-only collection (no search), 16 eval simulations

Pass criterion:
- Mean score clearly above random (>= 5.0 over 10 episodes)
- Search variants beat policy-only control (Variant c)
- Checkpoint continuity: all resume from Gate 1 checkpoint_best.pt at step 5,000

CPU core assignments are read from .remote (cpu_cores_a/b/c) so they can be
adjusted to fit whatever box is in use.
"""
import subprocess
import sys
import time
from remote_config import load_config

cfg = load_config()

GATE1_CKPT = cfg.remote_path(
    "results/100k_benchmark/gate1_poc_16sims/"
    "mcts_offpolicy_BreakoutNoFrameskip-v4_s42_1790152903/checkpoints/checkpoint_best.pt"
)


def main():
    print("--> 1. Syncing trainer to remote box...")
    subprocess.run(
        cfg.scp(
            "atari_qwen/training/train_mcts_offpolicy.py",
            cfg.remote_path("atari_qwen/training/train_mcts_offpolicy.py"),
        ),
        check=True,
    )
    print("✓ Synced train_mcts_offpolicy.py")

    print("\n--> 2. Verifying Gate 1 resume checkpoint exists on remote box...")
    res = cfg.run_ssh(f"test -f '{GATE1_CKPT}' && echo 'EXISTS' || echo 'MISSING'", check=False)
    if "EXISTS" not in res.stdout:
        print(f"✗ Gate 1 checkpoint missing at {GATE1_CKPT}!")
        sys.exit(1)
    print(f"✓ Gate 1 resume checkpoint verified: {GATE1_CKPT}")

    print(f"\n--> 3. Dispatching Gate 2 Trio in Parallel (cores {cfg.cpu_cores_a}, {cfg.cpu_cores_b}, {cfg.cpu_cores_c})...")

    base_args = (
        f"--env-id BreakoutNoFrameskip-v4 --total-steps 20000 --num-envs 8 "
        f"--eval-episodes 10 --eval-interval 5000 "
        f"--min-replay-size 1000 --seed 42 --resume-from {GATE1_CKPT} --start-step 5000"
    )

    # Variant A: 16 Sims
    cmd_a = (
        f"cd {cfg.workspace} && "
        f"taskset -c {cfg.cpu_cores_a} nohup {cfg.python()} atari_qwen/training/train_mcts_offpolicy.py "
        f"{base_args} --num-simulations 16 --eval-simulations 16 "
        f"--log-dir results/100k_benchmark/S050a_gate2_sim16 "
        "> /workspace/gate2_sim16.log 2>&1 &"
    )
    cfg.run_ssh(cmd_a, check=False)
    print(f"✓ Dispatched Variant (a): 16 simulations on cores {cfg.cpu_cores_a}")

    # Variant B: 32 Sims
    cmd_b = (
        f"cd {cfg.workspace} && "
        f"taskset -c {cfg.cpu_cores_b} nohup {cfg.python()} atari_qwen/training/train_mcts_offpolicy.py "
        f"{base_args} --num-simulations 32 --eval-simulations 32 "
        f"--log-dir results/100k_benchmark/S050b_gate2_sim32 "
        "> /workspace/gate2_sim32.log 2>&1 &"
    )
    cfg.run_ssh(cmd_b, check=False)
    print(f"✓ Dispatched Variant (b): 32 simulations on cores {cfg.cpu_cores_b}")

    # Variant C: Policy-Only Control
    cmd_c = (
        f"cd {cfg.workspace} && "
        f"taskset -c {cfg.cpu_cores_c} nohup {cfg.python()} atari_qwen/training/train_mcts_offpolicy.py "
        f"{base_args} --collect-with-policy-only --eval-simulations 16 "
        f"--log-dir results/100k_benchmark/S050c_gate2_policy_only "
        "> /workspace/gate2_policy_only.log 2>&1 &"
    )
    cfg.run_ssh(cmd_c, check=False)
    print(f"✓ Dispatched Variant (c): Policy-Only Control on cores {cfg.cpu_cores_c}")

    print("\n--> 4. Monitoring initial startup for all 3 runs...")
    time.sleep(10)
    for tag, log in [
        ("16 Sims",      "/workspace/gate2_sim16.log"),
        ("32 Sims",      "/workspace/gate2_sim32.log"),
        ("Policy Only",  "/workspace/gate2_policy_only.log"),
    ]:
        res = cfg.run_ssh(f"tail -n 8 {log} 2>/dev/null || true", check=False)
        print(f"\n--- [{tag}] ---")
        print(res.stdout.strip())


if __name__ == "__main__":
    main()
