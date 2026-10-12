import subprocess
from remote_config import load_config

cfg = load_config()

cmd = cfg.ssh(
    f"{cfg.python()} -c \""
    "import torch, glob; "
    f"p = glob.glob('{cfg.remote_path(\"results/100k_benchmark\")}/gate1_poc_16sims/**/checkpoint_latest.pt', recursive=True)[0]; "
    "ckpt = torch.load(p, map_location='cpu', weights_only=False); "
    "print('Step:', ckpt['step']); "
    "print('MCTS Score:', ckpt['mean_eval_mcts']); "
    "print('Raw Score:', ckpt['mean_eval_raw']); "
    "print('Best Eval:', ckpt['best_eval'])\""
)
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout)
if r.stderr:
    print("ERR:", r.stderr)
