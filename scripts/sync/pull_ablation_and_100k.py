import subprocess
import os

target_dir = r"E:\MThesis_EXP\imports\town01_ablation_and_100k_benchmarks_20260921"
os.makedirs(target_dir, exist_ok=True)
print(f"Target dir: {target_dir}")

print("--- PULLING CARLA ABLATION EVAL JSONS & CHECKPOINTS ---")
cmd_carla = [
    "ssh", "-p", "24525", "-o", "StrictHostKeyChecking=no", "root@ssh5.vast.ai",
    "tar -czf - -C /workspace bench2drive_out/town01_control_perroute bench2drive_out/town01_augmented_perroute checkpoints/wor_town01_augmented/best_model.pth checkpoints/wor_town01_control/best_model.pth 2>/dev/null"
]
p1 = subprocess.Popen(cmd_carla, stdout=subprocess.PIPE)
p2 = subprocess.Popen(["tar", "-xzf", "-", "-C", target_dir], stdin=p1.stdout)
p1.stdout.close()
p2.communicate()
print(f"CARLA pull complete, rc={p2.returncode}")

print("\n--- PULLING ATARI 100K BENCHMARK LOGS & CHECKPOINTS ---")
cmd_atari = [
    "ssh", "-p", "33163", "-o", "StrictHostKeyChecking=no", "root@ssh1.vast.ai",
    "tar -czf - -C /workspace MThesis/results/100k_benchmark 2>/dev/null"
]
p3 = subprocess.Popen(cmd_atari, stdout=subprocess.PIPE)
p4 = subprocess.Popen(["tar", "-xzf", "-", "-C", target_dir], stdin=p3.stdout)
p3.stdout.close()
p4.communicate()
print(f"Atari pull complete, rc={p4.returncode}")

# Count files
count = 0
for root, dirs, files in os.walk(target_dir):
    count += len(files)
print(f"Total files in {target_dir}: {count}")
