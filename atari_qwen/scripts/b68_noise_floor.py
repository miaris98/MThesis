"""How much do reanalysis targets depend on the Gumbel noise alone? (bounds the cost of reusing one search per state per target window)
Real checkpoint (qnowd seed 1, final 120k), 1,536 roots = 256 window starts x 6 consecutive states from its replay's second half."""
import os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)  # mixer_diag.py sits next to this file
import mixer_diag as md
from atari_qwen.mcts.gumbel_mcts import GumbelMCTS
torch.set_grad_enabled(False); torch.set_num_threads(8)
ck = md.FINAL + r"\b100k_qnowd_s1\checkpoints\checkpoint_env100000_upd120000.pt"
m, upd, env = md.build(ck)
a = torch.load(ck, map_location="cpu", weights_only=False)["args"]; a = a if isinstance(a, dict) else vars(a)
gamma, H = a["discount"] ** 4, a["lstm_horizon"]
frames, ep_start = md.npz_memmap(md.REPLAY, "frames.npy"), md.npz_memmap(md.REPLAY, "ep_start.npy")
T, E = ep_start.shape
rng = np.random.default_rng(1)
t0 = rng.integers(T // 2, T - 6, size=256); e0 = rng.integers(0, E, size=256)
t = (t0[:, None] + np.arange(6)).reshape(-1); e = np.repeat(e0, 6)
idx = np.maximum(t[:, None] + np.arange(-3, 1), np.asarray(ep_start[t, e])[:, None])
obs = torch.from_numpy(np.stack([np.asarray(frames[idx[i], e[i]]) for i in range(len(t))]).reshape(len(t), 12, 96, 96)).float().div_(255.0)
s, v, p = m.initial_inference(obs)
v_root = m.support.vector_to_scalar(v)
prior = p.softmax(-1).numpy()
def gumbel(seed):
    g = torch.Generator().manual_seed(seed)
    return -torch.log(-torch.log(torch.rand(len(t), 4, generator=g).clamp(1e-12, 1 - 1e-12)))
def run(n, seed):
    t1 = time.time()
    vv, pi, _ = GumbelMCTS(4, m.support, num_simulations=n, discount=gamma, lstm_horizon=H).search(
        m, s, v_root, p, g=(torch.zeros(len(t), 4) if seed is None else gumbel(seed)))
    return np.asarray(vv, np.float64), np.asarray(pi, np.float64), time.time() - t1
TV = lambda x, y: 0.5 * np.abs(x - y).sum(-1)
ref_v, ref_pi, dt = run(16, 1)
print(f"checkpoint upd {upd}, {len(t)} roots, 16-sim search on CPU {dt:.1f} s | mean |value| {np.abs(ref_v).mean():.3f}")
print(f"prior vs 16-sim target: TV mean {TV(prior, ref_pi).mean():.3f} median {np.median(TV(prior, ref_pi)):.3f}")
for name, (n, seed) in {"16 sims, other noise": (16, 2), "16 sims, other noise (b)": (16, 3), "16 sims, no noise": (16, None),
                        "8 sims, other noise": (8, 2), "4 sims, other noise": (4, 2), "32 sims, other noise": (32, 2)}.items():
    vv, pi, dt = run(n, seed)
    d = TV(ref_pi, pi)
    print(f"{name:26s}: policy TV mean {d.mean():.3f} median {np.median(d):.3f} p90 {np.quantile(d, 0.9):.3f} | top action agree "
          f"{(ref_pi.argmax(1) == pi.argmax(1)).mean():.1%} | value |diff| mean {np.abs(ref_v - vv).mean():.3f} | {dt:.1f} s")
