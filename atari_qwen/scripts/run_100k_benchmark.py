"""Official 100k Benchmark Runner for GTrXL + EfficientZero v2 on Atari Breakout.

Compares:
1. S047a_100k_standard: Standard on-policy GTrXL+EZ2 recipe (num_steps=128, 4 epochs, ~48 updates)
2. S047b_100k_sample_efficient: Enhanced gradient density (num_steps=64, 6 epochs, ~97 updates)

Both run with the S046c champion recipe:
- Kaiming normal init on visual encoder
- GRU gating on GTrXL blocks
- Reward prediction auxiliary loss (weight=1.0)
- Value prediction auxiliary loss (weight=0.25)
- Consistency loss disabled (weight=0.0)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

TOTAL_STEPS = 100_000

CONFIGS = {
    "S047a_100k_standard": dict(
        total_steps=TOTAL_STEPS,
        num_envs=16,
        num_steps=128,
        update_epochs=4,
        eval_interval_updates=5,
        cnn_kaiming_init=True,
        use_gru_gating=True,
        reward_loss_weight=1.0,
        ez_value_loss_weight=0.25,
        consistency_loss_weight=0.0,
    ),
    "S047b_100k_sample_efficient": dict(
        total_steps=TOTAL_STEPS,
        num_envs=16,
        num_steps=64,
        update_epochs=6,
        eval_interval_updates=10,
        cnn_kaiming_init=True,
        use_gru_gating=True,
        reward_loss_weight=1.0,
        ez_value_loss_weight=0.25,
        consistency_loss_weight=0.0,
    ),
    "S048a_100k_dense": dict(
        total_steps=TOTAL_STEPS,
        num_envs=16,
        num_steps=32,
        update_epochs=8,
        eval_interval_updates=20,
        cnn_kaiming_init=True,
        use_gru_gating=True,
        reward_loss_weight=1.0,
        ez_value_loss_weight=0.25,
        consistency_loss_weight=0.0,
    ),
    "S048b_100k_ultra_dense": dict(
        total_steps=TOTAL_STEPS,
        num_envs=16,
        num_steps=16,
        unroll_steps=3,
        update_epochs=8,
        eval_interval_updates=40,
        cnn_kaiming_init=True,
        use_gru_gating=True,
        reward_loss_weight=1.0,
        ez_value_loss_weight=0.25,
        consistency_loss_weight=0.0,
    ),
    "S048c_100k_hyper_dense": dict(
        total_steps=TOTAL_STEPS,
        num_envs=16,
        num_steps=16,
        unroll_steps=3,
        update_epochs=10,
        eval_interval_updates=40,
        cnn_kaiming_init=True,
        use_gru_gating=True,
        reward_loss_weight=1.0,
        ez_value_loss_weight=0.25,
        consistency_loss_weight=0.0,
    ),
    "S048d_100k_n8_e6": dict(
        total_steps=TOTAL_STEPS,
        num_envs=16,
        num_steps=8,
        unroll_steps=2,
        update_epochs=6,
        eval_interval_updates=80,
        cnn_kaiming_init=True,
        use_gru_gating=True,
        reward_loss_weight=1.0,
        ez_value_loss_weight=0.25,
        consistency_loss_weight=0.0,
    ),
    "S048e_100k_n32_e10": dict(
        total_steps=TOTAL_STEPS,
        num_envs=16,
        num_steps=32,
        unroll_steps=5,
        update_epochs=10,
        eval_interval_updates=20,
        cnn_kaiming_init=True,
        use_gru_gating=True,
        reward_loss_weight=1.0,
        ez_value_loss_weight=0.25,
        consistency_loss_weight=0.0,
    ),
}


def run_one(name: str, seed: int, cfg: dict, out_root: Path):
    run_name = f"{name}_s{seed}"
    log_dir = f"results/100k_benchmark/{run_name}"
    kwargs = dict(cfg, seed=seed, log_dir=log_dir)
    arg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    
    code = (
        "import os\n"
        "os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'\n"
        "from atari_qwen.training.train_gtrxl_ez2_onpolicy import train_onpolicy_gtrxl_ez2\n"
        f"train_onpolicy_gtrxl_ez2({arg_str})\n"
    )
    
    log_path = out_root / f"{run_name}.log"
    print(f"\n{'=' * 90}\n>>> STARTING: {run_name}\n>>> CONFIG: {arg_str}\n{'=' * 90}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run([sys.executable, "-c", code], stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f">>> {run_name} FINISHED rc={proc.returncode} in {elapsed / 60:.1f} min -> {log_path}", flush=True)
    return log_path, kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["S047a_100k_standard", "S047b_100k_sample_efficient"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--parallel", action="store_true", help="Launch variants in background concurrently")
    args = parser.parse_args()

    out_root = Path(REPO_ROOT) / "results" / "100k_benchmark" / "_logs"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.parallel:
        processes = []
        for name in args.variants:
            cfg = CONFIGS[name]
            for seed in args.seeds:
                run_name = f"{name}_s{seed}"
                log_dir = f"results/100k_benchmark/{run_name}"
                kwargs = dict(cfg, seed=seed, log_dir=log_dir)
                arg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                code = (
                    "import os\n"
                    "os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'\n"
                    "from atari_qwen.training.train_gtrxl_ez2_onpolicy import train_onpolicy_gtrxl_ez2\n"
                    f"train_onpolicy_gtrxl_ez2({arg_str})\n"
                )
                log_path = out_root / f"{run_name}.log"
                print(f">>> Spawning parallel process for {run_name} -> {log_path}", flush=True)
                fh = open(log_path, "w")
                p = subprocess.Popen([sys.executable, "-c", code], stdout=fh, stderr=subprocess.STDOUT)
                processes.append((run_name, p, fh, log_path))
        
        print(f">>> All {len(processes)} processes launched in parallel. Monitoring...", flush=True)
        for run_name, p, fh, log_path in processes:
            p.wait()
            fh.close()
            print(f">>> {run_name} completed with rc={p.returncode}", flush=True)
    else:
        for name in args.variants:
            cfg = CONFIGS[name]
            for seed in args.seeds:
                run_one(name, seed, cfg, out_root)


if __name__ == "__main__":
    main()
