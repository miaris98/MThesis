"""profile_step.py — Profile MCTS search latency on the remote box.

Uses subprocess+ssh so no local paramiko dependency is needed.
"""
import subprocess
from remote_config import load_config

cfg = load_config()

script = f"""
import time, torch, sys
sys.path.insert(0, '{cfg.workspace}')
from atari_qwen.training.train_mcts_offpolicy import make_vector_atari_envs
from atari_qwen.models.gtrxl_agent import ImpalaGTrXLAgent
from atari_qwen.mcts.mcts_engine import MCTSEngine

device = torch.device('cuda')
envs = make_vector_atari_envs('BreakoutNoFrameskip-v4', num_envs=8, seed=42)
agent = ImpalaGTrXLAgent(action_dim=4, in_channels=4, embed_dim=256, depth=4, num_heads=4, ffn_dim=1024, unroll_steps=5).to(device)
engine = MCTSEngine(action_dim=4, num_simulations=8)

obs, _ = envs.reset()
obs_norm = torch.as_tensor(obs, device=device).float() / 255.0

print("Profiling search_batch...")
t0 = time.time()
for _ in range(5):
    with torch.no_grad():
        latent_z, pol_repr = agent.encode_observation(obs_norm)
        probs, acts, vals = engine.search_batch(latent_z, agent, device, root_policy_reprs=pol_repr)
t1 = time.time()
print(f"5 search_batch calls took: {{t1 - t0:.3f}}s (avg: {{(t1 - t0)/5:.4f}}s per step)")
"""

proc = subprocess.run(
    cfg.ssh_T(f"{cfg.python()} -"),
    input=script.encode("utf-8"),
    capture_output=True,
)
print("STDOUT:")
print(proc.stdout.decode("utf-8", errors="replace"))
print("STDERR:")
print(proc.stderr.decode("utf-8", errors="replace"))
