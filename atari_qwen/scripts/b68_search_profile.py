"""TODO B68 (2026-10-26): where the 30m reanalysis search spends its time, on the GPU, as the trainer runs it.

Builds the 30m Qwen mixer model (QK-norm, random weights: the cost does not depend on the weight values), batch 1,536 roots, 16 sims,
the CUDA-graphed bf16 search (GraphedSearch, as --graph-search). Reports: ms per search; ms of one recurrent_inference at batch 1,536
and 6,144 (network share of a simulation step, and how much bigger batches help); an eager torch.profiler op split; a bf16 vs FP8
GEMM check at the FFN shape. Caps its own VRAM (--mem-frac) so it cannot push co-running jobs into OOM.
usage: python b68_search_profile.py [--mem-frac 0.15] [--roots 1536]"""
import argparse, time
import torch
from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport
from atari_qwen.mcts.gumbel_mcts import GumbelMCTS, GraphedSearch

ap = argparse.ArgumentParser()
ap.add_argument("--mem-frac", type=float, default=0.15)
ap.add_argument("--roots", type=int, default=1536)
ap.add_argument("--size", default="30m")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
dev, bf = torch.device("cuda"), torch.bfloat16
torch.manual_seed(0)
sup = DiscreteSupport()
m = EZV2Model(4, obs_channels=12, support=sup, trunk="qwen", mixer_size=args.size, mixer_qk_norm=True).to(dev).eval()
R = args.roots
obs = torch.rand(R, 12, 96, 96, device=dev)


def timed(fn, n=5):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return 1000 * sorted(ts)[len(ts) // 2]


with torch.no_grad(), torch.autocast("cuda", dtype=bf):
    s, v, p = m.initial_inference(obs)
    v = sup.vector_to_scalar(v)
gs = GraphedSearch(GumbelMCTS(4, sup, num_simulations=16, discount=0.997 ** 4, lstm_horizon=5), m, amp_dtype=bf)
t_search = timed(lambda: gs(s, v, p, True))
print(f"[B68 profile] {args.size} mixer, {R} roots, 16 sims, graphed bf16 search: {t_search:.1f} ms ({t_search / 16:.1f} ms per simulation step)")

with torch.no_grad(), torch.autocast("cuda", dtype=bf):
    for b in (R, 2 * R):
        try:
            sb = s.repeat(b // R, 1, 1, 1)
            a = torch.randint(0, 4, (b,), device=dev)
            h = (torch.zeros(1, b, m.lstm_hidden, device=dev), torch.zeros(1, b, m.lstm_hidden, device=dev))
            t_ri = timed(lambda: m.recurrent_inference(sb, a, h))
            t_mix = timed(lambda: m.dyn_mixer(sb))
            print(f"[B68 profile] recurrent_inference batch {b}: {t_ri:.1f} ms ({1000 * t_ri / b:.1f} us per state), dyn_mixer {t_mix:.1f} ms")
        except torch.OutOfMemoryError:
            print(f"[B68 profile] batch {b}: over this process's VRAM cap, skipped"); torch.cuda.empty_cache()
        del sb
    # op-level split of ONE dynamics mixer call at batch R (aten ops with their GPU time)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        m.dyn_mixer(s); torch.cuda.synchronize()
    ops = [e for e in prof.key_averages() if e.key.startswith("aten::") and e.self_device_time_total > 0]
    tot_mix = sum(e.self_device_time_total for e in ops)
    print(f"[B68 profile] one dyn_mixer call at batch {R}: {tot_mix / 1000:.1f} ms GPU, by aten op:")
    for e in sorted(ops, key=lambda e: -e.self_device_time_total)[:14]:
        print(f"    {100 * e.self_device_time_total / tot_mix:5.1f}%  {e.key:40s} calls {e.count}")

    mc = GumbelMCTS(4, sup, num_simulations=16, discount=0.997 ** 4, lstm_horizon=5)
    mc.search(m, s, v, p, add_noise=True); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        mc.search(m, s, v, p, add_noise=True); torch.cuda.synchronize()
    ev = prof.key_averages()
    tot = sum(e.self_device_time_total for e in ev)
    cats = {"gemm/matmul": ("gemm", "matmul", "mm", "addmm", "linear", "cutlass", "sm90", "sm80", "xmma", "ampere"),
            "attention": ("attention", "sdpa", "flash", "fmha", "efficient"),
            "conv/batchnorm": ("conv", "cudnn", "batch_norm", "bn_", "winograd", "implicit"),
            "lstm": ("lstm", "rnn"),
            "norm/elementwise": ("elementwise", "vectorized", "reduce", "pow", "rsqrt", "mean", "mul", "add", "silu", "copy", "cast")}
    split = {k: 0.0 for k in cats}; split["other (tree ops etc.)"] = 0.0
    for e in ev:
        n = e.key.lower()
        for k, keys in cats.items():
            if any(x in n for x in keys):
                split[k] += e.self_device_time_total; break
        else:
            split["other (tree ops etc.)"] += e.self_device_time_total
    print(f"[B68 profile] eager search GPU time {tot / 1000:.1f} ms; split: " + ", ".join(f"{k} {100 * x / tot:.0f}%" for k, x in split.items()))
    print("[B68 profile] top kernels:")
    for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:12]:
        print(f"    {100 * e.self_device_time_total / tot:5.1f}%  {e.key[:110]}")

# FP8 vs bf16 at the FFN shape (tokens x 512 @ 512 x 1536)
Mrows = R * 36
x = torch.randn(Mrows, 512, device=dev, dtype=bf); w = torch.randn(1536, 512, device=dev, dtype=bf)
t_bf = timed(lambda: x @ w.t(), n=20)
fl = 2 * Mrows * 512 * 1536
msg = f"[B68 profile] FFN-shape GEMM {Mrows}x512x1536: bf16 {t_bf:.2f} ms ({fl / t_bf / 1e9:.0f} TFLOPS)"
try:
    xf, wf = x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
    one = torch.ones((), device=dev)
    t_f8 = timed(lambda: torch._scaled_mm(xf, wf.t(), scale_a=one, scale_b=one, out_dtype=bf), n=20)
    msg += f"; fp8 e4m3 {t_f8:.2f} ms ({fl / t_f8 / 1e9:.0f} TFLOPS, x{t_bf / t_f8:.2f})"
except Exception as ex:
    msg += f"; fp8 not available here ({type(ex).__name__}: {str(ex)[:80]})"
print(msg)
print(f"[B68 profile] peak VRAM of this process: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
