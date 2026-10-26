"""Component diagnostics for the EZ-V2 champion (GTrXL, dead mixer) vs the Qwen-small 100k runs, on the SAME replay states.

Per checkpoint, on N states from the Qwen seed-1 replay (CPU, eval mode):
  - attention entropy of every mixer block (normalised by ln 36: 1 = uniform), its peak weight, and the largest |logit|
  - mixer contribution |out(norm_out(h))| / |t| on the representation and on one dynamics step (the replay's actions)
  - mean |alpha_attn| and |alpha_ffn| per Qwen block
  - policy prior entropy (nats) and P(FIRE), predicted value (support expectation, h-inverse)
usage: py -3 mixer_diag.py  (paths below; writes mixer_diag.json next to this file)"""
import json, math, os, re, sys, tarfile, zipfile
import numpy as np
import torch
import torch.nn.functional as F

REPO = r"C:\Users\miari\Desktop\MThesis"
sys.path.insert(0, REPO)
from atari_qwen.models.ez_model import EZV2Model, DiscreteSupport  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = r"E:\MThesis_EXP\live_20261025_final"
TAR = FINAL + r"\box5090\box5090_rest.tar"
CHAMP = r"E:\MThesis_EXP\live_20261011_atari_gtrxl_champions"
REPLAY = FINAL + r"\b100k_qnowd_s1\checkpoints\replay_latest.npz"
N, SEED = 192, 0
torch.set_grad_enabled(False)


def npz_memmap(path, name):
    """np.memmap of one member of an UNcompressed .npz (np.savez), without reading the whole file."""
    info = zipfile.ZipFile(path).getinfo(name)
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        start = info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little")
        f.seek(start)
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, ver)
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=off, order="F" if fortran else "C")


def states():
    frames, ep_start, act = (npz_memmap(REPLAY, k + ".npy") for k in ("frames", "ep_start", "act"))
    T, E = ep_start.shape
    rng = np.random.default_rng(SEED)
    t = rng.integers(T // 2, T, size=N)              # the second half: late-training states
    e = rng.integers(0, E, size=N)
    idx = np.maximum(t[:, None] + np.arange(-3, 1), np.asarray(ep_start[t, e])[:, None])
    f = np.stack([np.asarray(frames[idx[i], e[i]]) for i in range(N)])     # (N, 4, 3, 96, 96)
    obs = torch.from_numpy(f.reshape(N, 12, 96, 96)).float().div_(255.0)
    return obs, torch.as_tensor(np.asarray(act[t, e]), dtype=torch.long)


def build(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    if isinstance(a, dict):
        from types import SimpleNamespace
        a = SimpleNamespace(**a)
    m = EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk=a.trunk, norm=a.norm,
                  state_hw=int(np.ceil(a.frame_size / 16)), mixer_out_init_std=a.mixer_out_init_std,
                  mixer_bg_init=a.mixer_bg_init, simnorm=getattr(a, "simnorm", 0) or 0,
                  mixer_pos=getattr(a, "mixer_pos", "abs") or "abs", mixer_size=getattr(a, "mixer_size", None),
                  mixer_qk_norm=bool(getattr(a, "mixer_qk_norm", False)))
    m.load_state_dict(ck["model"])
    return m.eval(), ck.get("updates"), ck.get("env_steps")


def mixer_stats(mixer, x):
    """Manual pass through a Qwen TokenMixer: per-block attention stats, then the output/input norm ratio."""
    B, C, H, W = x.shape
    t = x.flatten(2).transpose(1, 2)
    h = mixer.inp(t) + mixer.pos
    blocks = []
    for blk in mixer.blocks:
        att, xn = blk.attn, blk.norm1(h)
        q = att.q_proj(xn).view(B, -1, att.num_heads, att.head_dim).transpose(1, 2)
        k = att.k_proj(xn).view(B, -1, att.num_heads, att.head_dim).transpose(1, 2)
        if getattr(att, "q_norm", None) is not None:  # TODO B67: the logits a QK-norm block actually uses
            q, k = att.q_norm(q), att.k_norm(k)
        s = (q @ k.transpose(-2, -1)) * att.scale
        p = s.softmax(-1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(-1)                     # (B, heads, N)
        blocks.append({"entropy": float(ent.mean() / math.log(t.shape[1])),
                       "entropy_min_head": float(ent.mean(dim=(0, 2)).min() / math.log(t.shape[1])),
                       "peak": float(p.amax(-1).mean()), "logit_absmax": float(s.abs().amax()),
                       "alpha_attn": float(blk.alpha_attn.abs().mean()), "alpha_ffn": float(blk.alpha_ffn.abs().mean())})
        h = blk(h)
    out = mixer.out(mixer.norm_out(h))
    return blocks, float(out.norm() / t.norm())


def diag(ck_path, obs, act):
    m, upd, env = build(ck_path)
    r = {"ckpt": ck_path, "updates": upd, "env_steps": env, "trunk": m.trunk}
    pre = m.repr_res(m.down(obs))
    s = m.representation(obs)
    if m.trunk == "qwen":
        r["repr_blocks"], r["repr_ratio"] = mixer_stats(m.repr_mixer, pre)
    elif m.trunk == "gtrxl":
        t = pre.flatten(2).transpose(1, 2)
        r["repr_ratio"] = float((s.flatten(2).transpose(1, 2) - t).norm() / t.norm())
    v, p = m.prediction(s)
    prob = p.softmax(-1)
    r["policy_entropy"] = float(-(prob * prob.clamp_min(1e-12).log()).sum(-1).mean())
    r["p_fire"] = float(prob[:, 1].mean())
    r["greedy_action_share"] = np.bincount(prob.argmax(-1).numpy(), minlength=4).tolist()
    sup = m.support
    vals = (v.softmax(-1) * torch.arange(sup.vmin, sup.vmax + 1)).sum(-1)
    r["value_mean"] = float(sup.h_inv(vals).mean())
    # one dynamics step with the replay's actions: the state just before the dynamics mixer
    plane = (act.float() / m.action_dim).view(-1, 1, 1, 1).expand(-1, 1, m.hw, m.hw)
    plane = F.relu(m.act_ln(m.act_conv(plane)))
    x = m.dyn_res(F.relu(m.dyn_bn(m.dyn_conv(torch.cat([s, plane], 1))) + s))
    if m.trunk == "qwen":
        r["dyn_blocks"], r["dyn_ratio"] = mixer_stats(m.dyn_mixer, x)
    elif m.trunk == "gtrxl":
        t = x.flatten(2).transpose(1, 2)
        r["dyn_ratio"] = float((m.dyn_mixer(x).flatten(2).transpose(1, 2) - t).norm() / t.norm())
    return r


def qwen_ckpts():
    """Extract the 10k-interval Qwen checkpoints from the box backup tar into the scratch folder (once)."""
    out = os.path.join(HERE, "ckpts")
    os.makedirs(out, exist_ok=True)
    want = re.compile(r"b100k/out/b100k_qnowd_s(\d)/checkpoints/checkpoint_env(\d+)_upd(\d+)\.pt$")
    paths = {}
    with tarfile.open(TAR) as tf:
        for mem in tf:
            mm = want.search(mem.name)
            if not mm or mm.group(3) == "120000":
                continue
            dst = os.path.join(out, f"s{mm.group(1)}_env{int(mm.group(2)):06d}_upd{int(mm.group(3)):06d}.pt")
            if not os.path.exists(dst) or os.path.getsize(dst) != mem.size:
                with tf.extractfile(mem) as src, open(dst, "wb") as fh:
                    fh.write(src.read())
            paths[(int(mm.group(1)), int(mm.group(3)))] = dst
    for s in (1, 2):
        paths[(s, 120000)] = FINAL + rf"\b100k_qnowd_s{s}\checkpoints\checkpoint_env100000_upd120000.pt"
    return dict(sorted(paths.items()))


if __name__ == "__main__":
    obs, act = states()
    rows = []
    for (s, upd), p in qwen_ckpts().items():
        r = diag(p, obs, act); r["run"] = f"qwen_s{s}"; rows.append(r)
        rb, db = r["repr_blocks"], r["dyn_blocks"]
        print(f"qwen_s{s} upd {upd:6d} | repr ent {rb[0]['entropy']:.3f}/{rb[1]['entropy']:.3f} peak {rb[0]['peak']:.2f}/{rb[1]['peak']:.2f} "
              f"ratio {r['repr_ratio']:.3f} | dyn ent {db[0]['entropy']:.3f}/{db[1]['entropy']:.3f} peak {db[0]['peak']:.2f}/{db[1]['peak']:.2f} "
              f"ratio {r['dyn_ratio']:.3f} | logit max {max(b['logit_absmax'] for b in rb + db):.1f} | alpha a/f {rb[0]['alpha_attn']:.2f}/{rb[0]['alpha_ffn']:.2f} "
              f"| pi H {r['policy_entropy']:.3f} P(FIRE) {r['p_fire']:.3f} greedy {r['greedy_action_share']} | V {r['value_mean']:.2f}", flush=True)
    for s, name in ((1, "checkpoint_latest.pt"), (2, "checkpoint_env100000_upd120000.pt"), (3, "checkpoint_latest.pt"),
                    (4, "checkpoint_env100000_upd120000.pt")):
        p = os.path.join(CHAMP, f"s{s}", name)
        if os.path.exists(p):
            r = diag(p, obs, act); r["run"] = f"champ_s{s}"; rows.append(r)
            print(f"champ_s{s} upd {r['updates']} | repr ratio {r['repr_ratio']:.4f} dyn ratio {r['dyn_ratio']:.4f} | pi H {r['policy_entropy']:.3f} "
                  f"P(FIRE) {r['p_fire']:.3f} greedy {r['greedy_action_share']} | V {r['value_mean']:.2f}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "mixer_diag.json"), "w"), indent=1)
    print("wrote mixer_diag.json")
