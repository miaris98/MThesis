"""eval_check.py — Quick remote evaluation of a saved policy checkpoint."""
import subprocess
import sys
import os

# Allow running from repo root: python scratch/eval_check.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "setup"))
from remote_config import load_config

cfg = load_config()

code = f"""
import sys
sys.path.insert(0, '{cfg.workspace}')
import torch
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.training.train_gtrxl_ez2_turbo import evaluate_agent

device = torch.device('cuda')
agent = ImpalaGTrXLAgent(action_dim=4, unroll_steps=5, bg_init=0.0).to(device)
ckpt_path = '{cfg.remote_path("results/100k_benchmark/S049a_mcts_sim20_s42/mcts_offpolicy_BreakoutNoFrameskip-v4_s42_1790078656/checkpoints/model_best.pt")}'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
agent.load_state_dict(ckpt)

score, std = evaluate_agent(agent, 'BreakoutNoFrameskip-v4', device, num_episodes=3)
print(f'Reactive policy score: {{score:.2f}} +/- {{std:.2f}}')
"""

proc = subprocess.run(
    cfg.ssh_T(f"{cfg.python()} -"),
    input=code.encode("utf-8"),
    capture_output=True,
)
print("STDOUT:", proc.stdout.decode("utf-8", errors="replace"))
print("STDERR:", proc.stderr.decode("utf-8", errors="replace"))
