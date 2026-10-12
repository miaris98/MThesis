import subprocess
import os

target_dir = r"E:\MThesis_EXP\imports\final_session_sync_20260921"
os.makedirs(target_dir, exist_ok=True)
print(f"Target dir: {target_dir}")

print("--- PULLING FINAL LIGHTZERO TELEMETRY & RUN LOGS ---")
cmd = [
    "ssh", "-p", "50576", "-o", "StrictHostKeyChecking=no", "root@70.48.44.13",
    "tar -czf - -C /workspace LightZero/data_efficientzero lightzero_breakout_100k_seed0.log 2>/dev/null"
]
p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
p2 = subprocess.Popen(["tar", "-xzf", "-", "-C", target_dir], stdin=p1.stdout)
p1.stdout.close()
p2.communicate()
print(f"LightZero final pull complete, rc={p2.returncode}")

count = sum(len(files) for _, _, files in os.walk(target_dir))
print(f"Total files in {target_dir}: {count}")
