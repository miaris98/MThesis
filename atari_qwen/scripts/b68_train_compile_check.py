"""TODO B68 E8 (2026-10-26): torch.compile of the online model's Qwen mixers for the training step (--compile-model).

The training step runs the same Qwen block as the search (initial inference with grad, 5 representation passes for the consistency
targets, 5 unroll steps with grad, backward), whose time is mostly separate elementwise kernels (RMSNorm in fp32, gates, casts).
nn.Module.compile() on repr_mixer and dyn_mixer fuses them forward and backward and keeps the state-dict keys.
1. Acceptance (same-maths rule, docs/ops/2026-10-26_reanalysis_cost_analysis.md 6.4): one training forward + backward on a real
   checkpoint and a real replay batch, in bf16 eager, bf16 compiled and an fp32 reference. The compiled loss and gradients must be no
   further from fp32 than the eager ones are.
2. Speed: forward + backward at --batch, eager vs compiled, and the host time to enqueue it (close to the total = launch-bound).
Fixed soft targets stand in for the search targets (the check is about the arithmetic, not the targets).
VRAM-capped (--mem-frac) so co-running jobs cannot OOM.
usage: python b68_train_compile_check.py RUN_DIR [--batch 32] [--mem-frac 0.12] [--size from-checkpoint|30m]"""
import argparse, copy, glob, os, re, time, zipfile
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--mem-frac", type=float, default=0.12)
ap.add_argument("--size", default="from-checkpoint", help="30m: random 30m weights (no 30m checkpoint yet) on the same real batch")
ap.add_argument("--reps", type=int, default=1, help="real batches to compare on (rounding errors vary from batch to batch)")
ap.add_argument("--emulate-casts", action="store_true", help="torch._inductor.config.emulate_precision_casts: round where eager rounds")
ap.add_argument("--skip-speed", action="store_true")
ap.add_argument("--speed-only", action="store_true", help="no fp32 reference (its activations would not fit beside co-running jobs "
                                                            "at batch 256; random 30m weights are too chaotic for the numerics anyway)")
args = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(args.mem_frac)
import torch._dynamo.config as dynamo_config
setattr(dynamo_config, "recompile_limit" if hasattr(dynamo_config, "recompile_limit") else "cache_size_limit", 64)   # as the trainer
if args.emulate_casts:
    import torch._inductor.config as inductor_config
    inductor_config.emulate_precision_casts = True
dev, B, K = torch.device("cuda"), args.batch, 5
sup = DiscreteSupport()


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
size = getattr(a, "mixer_size", None) if args.size == "from-checkpoint" else args.size
base = EZV2Model(4, obs_channels=12, support=sup, trunk=a.trunk, norm=a.norm, state_hw=int(np.ceil(a.frame_size / 16)),
                 simnorm=getattr(a, "simnorm", 0) or 0, mixer_pos=getattr(a, "mixer_pos", "abs") or "abs",
                 mixer_size=size, mixer_qk_norm=bool(getattr(a, "mixer_qk_norm", False)),
                 mixer_where=getattr(a, "mixer_where", "both") or "both")
if args.size == "from-checkpoint":
    base.load_state_dict(ck["model"])
else:
    torch.manual_seed(0)
    with torch.no_grad():                    # zero-initialised mixer outputs and heads: perturb so every output depends on the weights
        for p in base.parameters():
            p.add_(0.02 * torch.randn_like(p))
base = base.to(dev).to(memory_format=torch.channels_last).train()

# real batches: stacks of positions t..t+K from the run's replay, the actions taken there
rp = os.path.join(args.run_dir, "checkpoints", "replay_latest.npz")
frames, ep_start, act = npz_member(rp, "frames.npy"), npz_member(rp, "ep_start.npy"), npz_member(rp, "act.npy")
T, E = ep_start.shape


def make_batch(seed):
    global obs0, tgt_obs, acts, v_t, p_t, r_t
    rng = np.random.default_rng(seed)
    t0 = rng.integers(T // 2, T - K - 2, size=B); e0 = rng.integers(0, E, size=B)
    t = (t0[:, None] + np.arange(K + 1)).reshape(-1); e = np.repeat(e0, K + 1)
    idx = np.maximum(t[:, None] + np.arange(-3, 1), np.asarray(ep_start[t, e])[:, None])
    obs = torch.from_numpy(np.stack([np.asarray(frames[idx[i], e[i]]) for i in range(len(t))]).reshape(B, K + 1, 12, 96, 96))
    obs = obs.float().div_(255.0).to(dev)
    obs0 = obs[:, 0].contiguous(memory_format=torch.channels_last)
    tgt_obs = obs[:, 1:].reshape(B * K, 12, 96, 96).contiguous(memory_format=torch.channels_last).view(B, K, 12, 96, 96)
    acts = torch.as_tensor(np.asarray(act[t0[:, None] + np.arange(K), e0[:, None]]), device=dev)
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    v_t = sup.scalar_to_vector(5 * torch.rand(B, K + 1, device=dev, generator=g))
    p_t = torch.softmax(2 * torch.randn(B, K + 1, 4, device=dev, generator=g), -1)
    r_t = sup.scalar_to_vector(torch.rand(B, K, device=dev, generator=g).cumsum(1))


def train_loss(m, amp):
    """The trainer's training forward (train_ez_offpolicy.py, update section) on fixed targets."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        s, v_log, p_log = m.initial_inference(obs0)
        with torch.no_grad():
            tgt_proj = torch.stack([m.project(m.representation(tgt_obs[:, k]), with_grad=False).float() for k in range(K)], dim=1)
        ce = lambda logits, target_: -(F.log_softmax(logits.float(), -1) * target_).sum(-1)
        value_loss = 0.5 * ce(v_log, v_t[:, 0]); policy_loss = ce(p_log, p_t[:, 0])
        prefix_loss = torch.zeros(B, device=dev); cons_loss = torch.zeros(B, device=dev)
        hid = m.init_hidden(B, dev)
        for k in range(K):
            s, vp_log, v_log, p_log, hid = m.recurrent_inference(s, acts[:, k], hid)
            proj = m.project(s, with_grad=True).float()
            cons_loss = cons_loss - (F.normalize(proj, dim=-1, eps=1e-5) * F.normalize(tgt_proj[:, k], dim=-1, eps=1e-5)).sum(-1)
            prefix_loss = prefix_loss + ce(vp_log, r_t[:, k])
            value_loss = value_loss + 0.5 * ce(v_log, v_t[:, k + 1])
            policy_loss = policy_loss + ce(p_log, p_t[:, k + 1])
            if s.requires_grad:
                s.register_hook(lambda gr: gr * 0.5)
        return (prefix_loss + 0.25 * value_loss + policy_loss + 5.0 * cons_loss).mean() / K


def grads(m, amp):
    m.zero_grad(set_to_none=True)
    loss = train_loss(m, amp); loss.backward()
    gs = {"mixer": [], "rest": []}
    for n, q in m.named_parameters():
        if q.grad is not None:
            gs["mixer" if "mixer" in n else "rest"].append(q.grad.detach().float().flatten())
    return float(loss), {k: torch.cat(v) for k, v in gs.items() if v}


def compile_mixers(m):
    for mixer in (m.repr_mixer, m.dyn_mixer, m.pred_mixer):
        if mixer is not None:
            mixer.compile(dynamic=False)
    return m


m_f, m_e = copy.deepcopy(base), copy.deepcopy(base)       # train mode: BatchNorm normalises with batch statistics, so the running
m_c = compile_mixers(copy.deepcopy(base))                   # statistics each pass updates do not change later passes' outputs
rel = lambda x, y: float((x - y).norm() / y.norm().clamp_min(1e-12))
res = {"loss": [], "mixer": [], "rest": []}
if args.speed_only:
    del m_f
    make_batch(1)
for rep in range(0 if args.speed_only else args.reps):
    make_batch(1 + 10 * rep)
    lf, gf = grads(m_f, amp=False)                    # fp32 reference
    le, ge = grads(m_e, amp=True)
    t0_ = time.time(); lc, gc = grads(m_c, amp=True); t_comp = time.time() - t0_
    res["loss"].append((abs(le - lf), abs(lc - lf)))
    for k in gf:
        res[k].append((rel(ge[k], gf[k]), rel(gc[k], gf[k]), rel(gc[k], ge[k])))
    print(f"[B68 E8] {size} mixer, batch {B}, rep {rep}: loss fp32 {lf:.6f} | bf16 eager |d| {abs(le - lf):.2e} | bf16 compiled |d| "
          f"{abs(lc - lf):.2e} | grad rel. error vs fp32, eager / compiled: " +
          " ".join(f"{k} {res[k][-1][0]:.4f}/{res[k][-1][1]:.4f}" for k in gf) + (f" | first step incl. compile {t_comp:.0f} s" if rep == 0 else ""),
          flush=True)
casts = "emulated casts" if args.emulate_casts else "default casts"
for k, v in (res.items() if res["loss"] else ()):
    v = np.array(v)
    print(f"[B68 E8] {size}, {casts}, mean of {args.reps}: {k:5s} error vs fp32 eager {v[:, 0].mean():.4g} | compiled {v[:, 1].mean():.4g} "
          f"(ratio {v[:, 1].mean() / v[:, 0].mean():.2f}; compiled worse in {(v[:, 1] > v[:, 0]).sum()}/{len(v)})", flush=True)
if args.skip_speed:
    raise SystemExit


def step_time(m, n=5):
    def once():
        m.zero_grad(set_to_none=True); train_loss(m, True).backward()
    once(); torch.cuda.synchronize()
    tot, enq = [], []
    for _ in range(n):
        t_ = time.perf_counter(); once(); t1 = time.perf_counter(); torch.cuda.synchronize(); t2 = time.perf_counter()
        enq.append(t1 - t_); tot.append(t2 - t_)
    return 1000 * sorted(tot)[n // 2], 1000 * sorted(enq)[n // 2]


(te, qe), (tc, qc) = step_time(m_e), step_time(m_c)
print(f"[B68 E8] forward + backward at batch {B}: eager {te:.1f} ms (host enqueue {qe:.1f}) | compiled {tc:.1f} ms (host enqueue {qc:.1f}) "
      f"-> x{te / tc:.2f}", flush=True)
print(f"[B68 E8] peak VRAM of this process: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB (shared GPU: absolute times are inflated "
      f"by the co-running jobs, ratios hold)")
