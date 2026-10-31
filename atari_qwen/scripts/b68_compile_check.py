"""TODO B68 step E1 (2026-10-26): torch.compile of the target network's inference functions inside the CUDA-graphed reanalysis search.

1. Speed: 30m Qwen mixer (QK-norm, random weights), 1,536 roots, 16 sims, bf16 GraphedSearch with eager vs compiled
   initial_inference / recurrent_inference.
2. Acceptance (noise-floor rule, docs/ops/2026-10-26_reanalysis_cost_analysis.md 6.4): a real checkpoint and 1,536 real roots from its
   replay. Eager search under noise A vs noise B gives the floor; compiled vs eager under the same noise A must sit well inside it.
VRAM-capped (--mem-frac) so co-running jobs cannot OOM.
usage: python b68_compile_check.py RUN_DIR [--mem-frac 0.2] [--mode default]"""
import argparse, glob, os, re, time, zipfile
from types import SimpleNamespace
import numpy as np
import torch
from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport
from atari_qwen.mcts.gumbel_mcts import GumbelMCTS, GraphedSearch
from atari_qwen.training.reanalysis_cache import hashed_gumbel

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--mem-frac", type=float, default=0.2)
ap.add_argument("--mode", default="default")
ap.add_argument("--roots", type=int, default=1536)
ap.add_argument("--skip-speed", action="store_true")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
dev, bf, R = torch.device("cuda"), torch.bfloat16, args.roots
sup = DiscreteSupport()


def compile_inference(m):
    m.initial_inference = torch.compile(m.initial_inference, mode=args.mode, dynamic=False)
    m.recurrent_inference = torch.compile(m.recurrent_inference, mode=args.mode, dynamic=False)
    return m


def timed(fn, n=5):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return 1000 * sorted(ts)[len(ts) // 2]


def search_with(m, obs, g, amp=bf):
    """amp=None: the fp32 reference (no autocast anywhere)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=bf, enabled=amp is not None):
        s, v, p = m.initial_inference(obs)
        v = sup.vector_to_scalar(v)
    gs = GraphedSearch(GumbelMCTS(4, sup, num_simulations=16, discount=0.997 ** 4, lstm_horizon=5), m, amp_dtype=amp)
    out = gs(s, v, p, g=g)
    return out, (lambda: gs(s, v, p, g=g))


# 1. speed, 30m
if not args.skip_speed:
  torch.manual_seed(0)
  m30 = EZV2Model(4, obs_channels=12, support=sup, trunk="qwen", mixer_size="30m", mixer_qk_norm=True).to(dev).eval()
  obs = torch.rand(R, 12, 96, 96, device=dev)
  g = hashed_gumbel(np.arange(R), 0, 4, dev)
  _, run_eager = search_with(m30, obs, g)
  t_e = timed(run_eager)
  t0 = time.time(); compile_inference(m30); _, run_comp = search_with(m30, obs, g); t_build = time.time() - t0
  t_c = timed(run_comp)
  print(f"[B68 E1] 30m graphed search, {R} roots: eager {t_e:.0f} ms, compiled ({args.mode}) {t_c:.0f} ms -> x{t_e / t_c:.2f} "
        f"(compile + capture {t_build:.0f} s)")
  del m30, run_eager, run_comp; torch.cuda.empty_cache()


# 2. acceptance on a real checkpoint and real roots
def npz_member(path, name):
    """np.memmap of one member of an uncompressed .npz using numpy's public header readers (works on numpy 2.x)."""
    info = zipfile.ZipFile(path).getinfo(name)
    with open(path, "rb") as f:
        f.seek(info.header_offset); h = f.read(30)
        f.seek(info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0) else np.lib.format.read_array_header_2_0(f))
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=off, order="F" if fortran else "C")


cks = sorted(glob.glob(os.path.join(args.run_dir, "checkpoints", "checkpoint_env*_upd*.pt")),
             key=lambda p: int(re.search(r"upd(\d+)", p).group(1)))
ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
a = ck["args"]; a = SimpleNamespace(**a) if isinstance(a, dict) else a


def build():
    m = EZV2Model(4, obs_channels=12, support=sup, trunk=a.trunk, norm=a.norm, state_hw=int(np.ceil(a.frame_size / 16)),
                  simnorm=getattr(a, "simnorm", 0) or 0, mixer_pos=getattr(a, "mixer_pos", "abs") or "abs",
                  mixer_size=getattr(a, "mixer_size", None), mixer_qk_norm=bool(getattr(a, "mixer_qk_norm", False)),
                  mixer_where=getattr(a, "mixer_where", "both") or "both")
    m.load_state_dict(ck["model"]); return m.to(dev).eval()


rp = os.path.join(args.run_dir, "checkpoints", "replay_latest.npz")
frames, ep_start = npz_member(rp, "frames.npy"), npz_member(rp, "ep_start.npy")
T, E = ep_start.shape
rng = np.random.default_rng(1)
t0_ = rng.integers(T // 2, T - 6, size=R // 6); e0 = rng.integers(0, E, size=R // 6)
t = (t0_[:, None] + np.arange(6)).reshape(-1); e = np.repeat(e0, 6)
idx = np.maximum(t[:, None] + np.arange(-3, 1), np.asarray(ep_start[t, e])[:, None])
obs = torch.from_numpy(np.stack([np.asarray(frames[idx[i], e[i]]) for i in range(len(t))]).reshape(len(t), 12, 96, 96)).float().div_(255.0).to(dev)
keys = t * E + e
gA, gB = hashed_gumbel(keys, 1, 4, dev), hashed_gumbel(keys, 2, 4, dev)
me = build()
(vA, pA, _), _ = search_with(me, obs, gA)
(vB, pB, _), _ = search_with(me, obs, gB)
mc = compile_inference(build())
(vC, pC, _), _ = search_with(mc, obs, gA)
TV = lambda x, y: 0.5 * np.abs(x - y).sum(-1)
(vF, pF, _), _ = search_with(build(), obs, gA, amp=None)          # exact-maths reference: fp32, same noise as A
rows = {"bf16 eager vs bf16 eager, other noise (noise floor)": (pA, vA, pB, vB),
        "bf16 eager vs fp32, same noise (today's rounding)": (pF, vF, pA, vA),
        "bf16 compiled vs fp32, same noise": (pF, vF, pC, vC),
        "bf16 compiled vs bf16 eager, same noise": (pA, vA, pC, vC)}
for name, (p1, v1, p2, v2) in rows.items():
    d = TV(p1, p2)
    print(f"[B68 E1] {name:52s}: policy TV mean {d.mean():.4f} median {np.median(d):.4f} p90 "
          f"{np.quantile(d, 0.9):.4f} | top action agree {(p1.argmax(1) == p2.argmax(1)).mean():.1%} | value |diff| {np.abs(v1 - v2).mean():.4f}")
print(f"[B68 E1] peak VRAM of this process: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
