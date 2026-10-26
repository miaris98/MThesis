"""TODO B68 (2026-10-26): how much of the 30m dynamics mixer's time is removable by fusing its elementwise work (same maths).

The op profile (b68_search_profile.py) put ~67% of a dyn_mixer call in elementwise kernels (RMSNorm's fp32 upcasts, pow/mean/mul,
dtype copies, fp32 alpha gates and positions promoting the bf16 stream) and ~29% in the matmuls. Variants, batch 1,536 roots:
  eager            the code as it runs today
  F.rms_norm       RMSNorm.forward replaced by torch's native rms_norm (one fused kernel, fp32 accumulation, bf16 out)
  torch.compile    the whole mixer through Inductor (fuses the elementwise chains; matmuls stay cuBLAS)
Reports ms per call and the max |difference| of each output against eager (rounding-level only if the maths is the same).
usage: python b68_fusion_bench.py [--mem-frac 0.15]"""
import argparse, time
import torch
import torch.nn.functional as F
from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport
from src.models.transformer import layers

ap = argparse.ArgumentParser()
ap.add_argument("--mem-frac", type=float, default=0.15)
ap.add_argument("--roots", type=int, default=1536)
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
dev, bf = torch.device("cuda"), torch.bfloat16
torch.manual_seed(0)
m = EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk="qwen", mixer_size="30m", mixer_qk_norm=True).to(dev).eval()
with torch.no_grad():  # give the zero-initialised output projection and the gates real values, so differences are visible
    torch.nn.init.normal_(m.dyn_mixer.out.weight, std=0.02)
x = torch.randn(args.roots, 64, 6, 6, device=dev)


def timed(fn, n=10):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return 1000 * sorted(ts)[len(ts) // 2]


with torch.no_grad(), torch.autocast("cuda", dtype=bf):
    ref = m.dyn_mixer(x).float()
    t_eager = timed(lambda: m.dyn_mixer(x))
    print(f"[B68 fusion] eager dyn_mixer: {t_eager:.1f} ms")

    orig = layers.RMSNorm.forward
    layers.RMSNorm.forward = lambda self, h: F.rms_norm(h, (h.shape[-1],), self.weight.to(h.dtype), self.eps)
    y = m.dyn_mixer(x).float()
    t_rms = timed(lambda: m.dyn_mixer(x))
    print(f"[B68 fusion] F.rms_norm: {t_rms:.1f} ms (x{t_eager / t_rms:.2f}); max |diff| vs eager {(y - ref).abs().max():.2e} "
          f"(output scale {ref.abs().mean():.2e})")
    layers.RMSNorm.forward = orig

    t0 = time.time()
    cm = torch.compile(m.dyn_mixer, mode="max-autotune-no-cudagraphs", dynamic=False)
    y = cm(x).float(); torch.cuda.synchronize()
    print(f"[B68 fusion] torch.compile build: {time.time() - t0:.0f} s")
    t_comp = timed(lambda: cm(x))
    print(f"[B68 fusion] torch.compile: {t_comp:.1f} ms (x{t_eager / t_comp:.2f}); max |diff| vs eager {(y - ref).abs().max():.2e}")
print(f"[B68 fusion] peak VRAM of this process: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
