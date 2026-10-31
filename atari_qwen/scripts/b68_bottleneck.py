"""TODO B68 (2026-10-26): what an update's search costs once the cache and the compiled search are on, at the cache's miss sizes.

With the cache an update searches only its misses, padded to a bucket (96 ... 1,536). Within a target window the miss count falls from
~1,536 (right after a refresh) to tens, so most searches are small, and the part of a simulation step that does not shrink with the batch
(tree bookkeeping, small-kernel latency) decides what they cost. For each bucket: the graphed bf16 search with the eager and with the
compiled recurrent_inference (as --compile-search), one CUDA-graphed recurrent_inference call (the network's share of a step) and one
eager initial_inference (the cache's root / bootstrap pass on misses). Then the average search cost per update over a 100k run, with
misses searched per update (today) or merged over the 4 updates that share an env step. Random 30m weights: the cost does not depend
on the weight values. Caps its own VRAM (--mem-frac) so co-running jobs cannot OOM.
usage: python b68_bottleneck.py [--mem-frac 0.12] [--size 30m] [--buckets 96,192,384,768,1536]"""
import argparse, copy, math, time
import numpy as np
import torch
from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport
from atari_qwen.mcts.gumbel_mcts import GumbelMCTS, GraphedSearch
from atari_qwen.training.reanalysis_cache import BUCKETS, bucket

ap = argparse.ArgumentParser()
ap.add_argument("--mem-frac", type=float, default=0.12)
ap.add_argument("--size", default="30m")
ap.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
ap.add_argument("--fallback", default="1536:967,516",
                help="bucket:eager,compiled search ms to use where a bucket does not fit the VRAM cap (default: the E1 measurement)")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
import torch._dynamo.config as dynamo_config
# 5 buckets x (standalone graph + inside the search's capture) = 10 compiled variants; past the default limit of 8 the rest silently
# ran eagerly (the first two runs of this script: the 1,536 bucket's "compiled" time was the eager one)
setattr(dynamo_config, "recompile_limit" if hasattr(dynamo_config, "recompile_limit") else "cache_size_limit", 64)
dev, bf = torch.device("cuda"), torch.bfloat16
torch.manual_seed(0)
sup = DiscreteSupport()
m_e = EZV2Model(4, obs_channels=12, support=sup, trunk="qwen", mixer_size=args.size, mixer_qk_norm=True).to(dev).eval()
m_c = copy.deepcopy(m_e)
m_c.recurrent_inference = torch.compile(m_c.recurrent_inference, dynamic=False)


def timed(fn, n=7):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return 1000 * sorted(ts)[len(ts) // 2]


def graph_of(fn):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


rows = {}
fallback = {int(b): tuple(map(float, c.split(","))) for b, c in (x.split(":") for x in args.fallback.split(";") if x)}
with torch.no_grad(), torch.autocast("cuda", dtype=bf):
    for R in map(int, args.buckets.split(",")):
        try:
            obs = torch.rand(R, 12, 96, 96, device=dev)
            t_ii = timed(lambda: m_e.initial_inference(obs))
            s, v, p = m_e.initial_inference(obs)
            v = sup.vector_to_scalar(v)
            del obs                                   # 680 MB at 1,536 roots; the search needs only the latents
            torch.cuda.empty_cache()
            a = torch.randint(0, 4, (R,), device=dev)
            h = (torch.zeros(1, R, m_e.lstm_hidden, device=dev), torch.zeros(1, R, m_e.lstm_hidden, device=dev))
            out = {}
            for name, m in (("eager", m_e), ("compiled", m_c)):
                g = graph_of(lambda: m.recurrent_inference(s, a, h))
                t_ri = timed(g.replay, n=11)
                del g
                gs = GraphedSearch(GumbelMCTS(4, sup, num_simulations=16, discount=0.997 ** 4, lstm_horizon=5), m, amp_dtype=bf)
                t_s = timed(lambda: gs(s, v, p, True))
                del gs
                torch.cuda.empty_cache()
                out[name] = (t_s, t_ri)
        except torch.OutOfMemoryError:
            g = gs = None; torch.cuda.empty_cache()
            if R not in fallback:
                raise
            out = {"eager": (fallback[R][0], float("nan")), "compiled": (fallback[R][1], float("nan"))}
            rows[R] = (float("nan"), out)
            print(f"[B68 bottleneck] {R:5d} roots | over this process's VRAM cap: using the fallback search times {fallback[R]}", flush=True)
            continue
        rows[R] = (t_ii, out)
        (se, re_), (sc, rc) = out["eager"], out["compiled"]
        print(f"[B68 bottleneck] {R:5d} roots | search eager {se:7.1f} ms (network {16 * re_:6.1f}, rest {se - 16 * re_:6.1f}) | "
              f"compiled {sc:7.1f} ms (network {16 * rc:6.1f}, rest {sc - 16 * rc:6.1f}) | x{se / sc:.2f} | "
              f"initial_inference {t_ii:5.1f} ms", flush=True)

# fixed + per-root fit of each search (ms per search = c0 + c1 * roots)
Rs = np.array(sorted(rows), float)
fit = {}
for name in ("eager", "compiled"):
    ys = np.array([rows[int(R)][1][name][0] for R in Rs])
    c1, c0 = np.polyfit(Rs, ys, 1)
    fit[name] = (c0, c1)
    print(f"[B68 bottleneck] {name}: search ~ {c0:.1f} ms fixed + {1000 * c1:.1f} us per root ({c0 / 16:.2f} ms fixed per simulation step)")

# average search cost per update over a 100k run (E = 4, replay ratio 1: 98k updates while the buffer grows, then 22k on the full buffer).
# Misses at update j of a window: R * exp(-R j / N) (independent draws; the trainer's sampler matched this law, b68_cache_sim.py).
R0, W = 1536, 200
cost = {name: {R: rows[R][1][name][0] for R in rows} for name in ("eager", "compiled")}


def search_ms(n, name):
    if n < 1:
        return 0.0
    full, rem = divmod(int(math.ceil(n)), BUCKETS[-1])
    return full * cost[name][BUCKETS[-1]] + (cost[name][bucket(rem)] if rem else 0.0)


for name in ("eager", "compiled"):
    tot_u = tot_b = 0.0; n_upd = 0; miss_sum = 0.0
    for u in range(0, 120000, 4):
        N = min(100000, 2000 + u)
        js = [(u + i) % W for i in range(4)]
        ms = [R0 * math.exp(-R0 * j / N) for j in js]
        tot_u += sum(search_ms(x, name) for x in ms)
        tot_b += search_ms(sum(ms), name)
        miss_sum += sum(ms); n_upd += 4
    print(f"[B68 bottleneck] {name}: mean misses per update {miss_sum / n_upd:.0f}; search per update, misses searched per update "
          f"{tot_u / n_upd:.1f} ms, merged over the 4 updates of an env step {tot_b / n_upd:.1f} ms", flush=True)
print(f"[B68 bottleneck] peak VRAM of this process: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB (shared GPU: absolute times "
      f"are inflated by the co-running jobs, ratios hold)")
