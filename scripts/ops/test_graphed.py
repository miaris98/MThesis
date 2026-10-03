"""B23: GraphedSearch vs eager search on real Breakout states: exact equality + speed. usage: test_graphed.py CKPT"""
import sys, time, pathlib
sys.path.insert(0, ".")
import numpy as np, torch
from atari_qwen.training.eval_ez_checkpoints import load_model
from atari_qwen.training.train_ez_offpolicy import to_input, make_vector_atari_envs
from atari_qwen.mcts.gumbel_mcts import GraphedSearch

dev = torch.device("cuda"); ck = pathlib.Path(sys.argv[1]); steps = int(sys.argv[2]) if len(sys.argv) > 2 else 150
for sims, bound in ([(int(x.split(":")[0]), int(x.split(":")[1])) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else ((16, 6), (64, 10))):
    _, a, model, mcts = load_model(ck, dev, sims); model.eval()
    gs = GraphedSearch(mcts, model, depth_bound=bound)
    envs = make_vector_atari_envs(a.env_id, num_envs=30, seed=10_000, clip_reward=False, episodic_life=False,
                                  frame_size=a.frame_size, grayscale=False, fire_reset=False, max_episode_steps=27000)
    obs, _ = envs.reset(seed=10_000)
    bad = {"act": 0, "pol": 0, "val": 0}; t_e = t_g = 0.0; maxdiff = 0.0
    for t in range(steps):
        x = to_input(obs, dev)
        with torch.no_grad():
            s, v, p = model.initial_inference(x); v = model.support.vector_to_scalar(v)
        for noise in (False, True):
            torch.manual_seed(t); torch.cuda.synchronize(); t0 = time.perf_counter()
            re = mcts.search(model, s, v, p, add_noise=noise)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            torch.manual_seed(t); rg = gs(s, v, p, noise)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            if t >= 5 and not noise:
                t_e += t1 - t0; t_g += t2 - t1
            bad["act"] += not np.array_equal(re[2], rg[2])
            bad["pol"] += not np.array_equal(re[1], rg[1])
            bad["val"] += not np.array_equal(re[0], rg[0])
            maxdiff = max(maxdiff, float(np.abs(re[1] - rg[1]).max()))
        obs, *_ = envs.step(re[2])
    n = steps - 5
    print(f"sims {sims} bound {bound}: {steps} steps x 30 roots x {{no noise, noise}}: mismatching batches "
          f"actions {bad['act']} policies {bad['pol']} values {bad['val']}, max |policy diff| {maxdiff:.2e}; "
          f"fallbacks {gs.fallbacks}/{gs.calls}; eager {1000 * t_e / n:.0f} ms, graphed {1000 * t_g / n:.0f} ms per search "
          f"({t_e / t_g:.1f}x)", flush=True)
    envs.close()
