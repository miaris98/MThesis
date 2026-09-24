import subprocess

script = """
import json, glob, os

for path in sorted(glob.glob('/workspace/bench2drive_out/sent_*.json')):
    try:
        with open(path) as f:
            d = json.load(f)
        records = d.get('_checkpoint', {}).get('records', [])
        scores = [r.get('scores', {}).get('score_composed', 0.0) for r in records]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{os.path.basename(path)}: {len(scores)} routes | avg: {avg:.1f} | scores: {scores}")
    except Exception as e:
        print(f"{path}: {e}")
"""

res = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', '-p', '37276', 'root@85.10.218.46', '/venv/main/bin/python', '-'],
    input=script,
    text=True,
    capture_output=True
)
print("STDOUT:\n", res.stdout)
if res.stderr:
    print("STDERR:\n", res.stderr)
