import subprocess
import os

target_dir = r"E:\MThesis_EXP\imports\town01_ablation_and_100k_benchmarks_20260921"
os.makedirs(target_dir, exist_ok=True)

print(f"Target directory: {target_dir}")
print("Testing direct IP (70.48.44.13:50576) pull...")

cmd = [
    "ssh", "-p", "50576", "-o", "StrictHostKeyChecking=no", "root@70.48.44.13",
    "tar -czf - -C /workspace MThesis/results/100k_benchmark"
]

p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
p2 = subprocess.Popen(["tar", "-xzf", "-", "-C", target_dir], stdin=p1.stdout)
p1.stdout.close()
p2.communicate()

print(f"Direct IP tar pull finished with rc = {p2.returncode}")

# Count files
count = sum(len(files) for _, _, files in os.walk(os.path.join(target_dir, "MThesis", "results", "100k_benchmark")))
print(f"Total synced files in 100k_benchmark: {count}")
