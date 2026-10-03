"""Max / percentile tree depth of the eager search on real Breakout states (picks GraphedSearch's depth bound)."""
import sys, pathlib
sys.path.insert(0, ".")
import numpy as np, torch
from atari_qwen.training.eval_ez_checkpoints import load_model
from atari_qwen.training.train_ez_offpolicy import to_input, make_vector_atari_envs
dev = torch.device("cuda"); ck = pathlib.Path(sys.argv[1])
for sims in (16, 32, 64):
    _, a, model, mcts = load_model(ck, dev, sims); model.eval()
    envs = make_vector_atari_envs(a.env_id, num_envs=30, seed=10_000, clip_reward=False, episodic_life=False,
                                  frame_size=a.frame_size, grayscale=False, fire_reset=False, max_episode_steps=27000)
    obs, _ = envs.reset(seed=10_000); depths = []
    for t in range(60):
        x = to_input(obs, dev)
        with torch.no_grad():
            s, v, p = model.initial_inference(x); v = model.support.vector_to_scalar(v)
            _, _, act = mcts.search(model, s, v, p, add_noise=False)
        depths.append(mcts.depth.max(1).values.cpu().numpy())   # per-tree max depth
        obs, *_ = envs.step(act)
    d = np.concatenate(depths)
    print(f"sims {sims}: tree max depth over {len(d)} trees: mean {d.mean():.1f} p50 {np.percentile(d, 50):.0f} p99 {np.percentile(d, 99):.0f} max {d.max()}", flush=True)
    envs.close()
