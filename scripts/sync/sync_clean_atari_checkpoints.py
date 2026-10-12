import subprocess
import os

target_dir = r"E:\MThesis_EXP\imports\town01_ablation_and_100k_benchmarks_20260921\100k_benchmark"
os.makedirs(target_dir, exist_ok=True)

print(f"Target directory: {target_dir}")
print("Syncing 100k benchmark results via scp...")

scp_cmd = [
    "scp", "-P", "33163", "-r", "-o", "StrictHostKeyChecking=no",
    "root@ssh1.vast.ai:/workspace/MThesis/results/100k_benchmark/*",
    target_dir
]

res = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
print(f"SCP Return Code: {res.returncode}")
if res.stderr:
    print(f"SCP Stderr: {res.stderr[:500]}")

# Count files
count = sum(len(files) for _, _, files in os.walk(target_dir))
print(f"Total synced files in {target_dir}: {count}")
