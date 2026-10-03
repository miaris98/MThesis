#!/usr/bin/env python3
"""TODO B15 (plasticity diagnostics) and B18 step 1 (GTrXL attention entropy) on saved checkpoints, CPU only.

For every checkpoint, on one fixed set of states (4-frame stacks cut from a replay buffer, so all checkpoints see the same
inputs), report:

  B15  * dormant-neuron ratio (tau = 0.025, Sokar et al. 2023): a neuron is dormant when its mean |post-activation| over
         the states, divided by the layer's mean, is <= tau; per layer group (conv trunk / dynamics / prediction / reward
         head / GTrXL FFN), as the share of dormant neurons
       * srank (Kumar et al. 2021, delta = 0.01) of the representation state (64 x 6 x 6 = 2304 features) and of the
         policy/value head input, and the same normalised by min(n_states, n_features)
       * weight norms (total, and per top-level module)
  B18  * (GTrXL only) per mixer block (repr / dyn x layer): attention entropy / log(36) averaged over states, queries
         and heads, per head; mean peak attention weight; mean gate opening z of the two GRU gates; relative change the
         whole mixer makes to the state (|t_out - t| / |t|)

The gradient norm of B15 needs the training loss and is not computed here.

    py scripts/analysis/b15_b18_diagnostics.py --replay E:/.../replay_latest.npz --out diag.json CKPT [CKPT ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atari_qwen.models.ez_model import DiscreteSupport, EZV2Model  # noqa: E402

TAU, DELTA = 0.025, 0.01


def stacks_from_replay(path: str, n: int, seed: int = 0) -> torch.Tensor:
    """n 4-frame stacks (12, 96, 96) in [0, 1], oldest frame first, from replay frames (T, env, 3, H, W)."""
    z = np.load(path, mmap_mode="r")
    frames, ep_start = z["frames"], z["ep_start"]
    T, E = frames.shape[:2]
    rnd = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        t, e = int(rnd.integers(3, T)), int(rnd.integers(0, E))
        if (ep_start[t - 3:t + 1, e] != ep_start[t, e]).any():  # a stack must not cross an episode start
            continue
        out.append(np.concatenate([frames[t - 3 + k, e] for k in range(4)], axis=0))
    return torch.from_numpy(np.stack(out)).float().div_(255.0)


def load(ck_path: str):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    A = ck["model"]["p_fc.3.weight"].shape[0] if "p_fc.3.weight" in ck["model"] else 4
    model = EZV2Model(A, obs_channels=12, support=DiscreteSupport(), trunk=a.trunk, norm=a.norm,
                      state_hw=int(math.ceil(a.frame_size / 16)))
    model.load_state_dict(ck["model"])
    return model.eval(), a


def group_of(name: str) -> str:
    if "mixer" in name:
        return "gtrxl_ffn"
    for key, grp in (("down", "conv_trunk"), ("repr_res", "conv_trunk"), ("dyn_", "dynamics"), ("act_", "dynamics"),
                     ("pred_res", "prediction"), ("v_", "prediction"), ("p_", "prediction"),
                     ("rew_", "reward_head"), ("proj", "projection")):
        if name.startswith(key):
            return grp
    return "other"


def dormant_stats(model, x, a_onehot=None):
    """Per layer group: dormant share (tau) of neurons; neurons = channels (conv) or units (linear)."""
    acts = {}
    hooks = []

    def keep(name, fn):
        def hook(_m, _i, out):
            acts[name] = fn(out).detach()
        return hook

    for name, m in model.named_modules():
        if name.startswith("proj"):
            continue
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            hooks.append(m.register_forward_hook(keep(name, lambda o: F.relu(o).abs().mean((0, 2, 3)))))
        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)) and not name.startswith("lstm"):
            act = F.relu if name == "rew_bn2" else F.elu  # the MLP heads use ELU, the LSTM output ReLU
            hooks.append(m.register_forward_hook(keep(name, lambda o, act=act: act(o).abs().mean(tuple(range(o.dim() - 1))))))
        elif name.endswith("w_gate"):  # SwiGLU gate of the GTrXL FFN
            hooks.append(m.register_forward_hook(keep(name, lambda o: F.silu(o).abs().mean((0, 1)))))
    with torch.no_grad():
        s, _, _ = model.initial_inference(x)
        a = torch.randint(0, model.action_dim, (x.shape[0],)) if a_onehot is None else a_onehot
        model.recurrent_inference(s, a, model.init_hidden(x.shape[0], x.device))
    for h in hooks:
        h.remove()
    groups = {}
    for name, v in acts.items():
        score = v / v.mean().clamp(min=1e-12)
        groups.setdefault(group_of(name), []).append((score <= TAU))
    return {g: float(torch.cat(v).float().mean()) for g, v in groups.items()}


def srank(feat: torch.Tensor) -> tuple:
    sv = torch.linalg.svdvals(feat)
    c = torch.cumsum(sv, 0) / sv.sum()
    k = int((c < 1 - DELTA).sum().item()) + 1
    return k, k / min(feat.shape)


def mixer_report(mixer, t_in):
    """Attention entropy / peak / gate opening per block of a TokenMixer, replaying its forward on t_in (B, C, H, W)."""
    B, C, H, W = t_in.shape
    t = t_in.flatten(2).transpose(1, 2)
    h = mixer.inp(t) + mixer.pos
    out = []
    for blk in mixer.blocks:
        x = blk.norm1(h)
        q = blk.q_proj(x).view(B, -1, blk.num_heads, blk.head_dim).transpose(1, 2)
        k = blk.k_proj(x).view(B, -1, blk.num_heads, blk.head_dim).transpose(1, 2)
        w = torch.softmax(q @ k.transpose(-2, -1) * blk.scale, dim=-1)                  # (B, heads, N, N)
        ent = -(w * w.clamp(min=1e-12).log()).sum(-1) / math.log(w.shape[-1])            # (B, heads, N)
        attn_out = blk._attention(x)
        z1 = torch.sigmoid(blk.gate1.w_z(h) + blk.gate1.u_z(attn_out) - blk.gate1.bg)
        h1 = blk.gate1(h, attn_out)
        z2 = torch.sigmoid(blk.gate2.w_z(h1) + blk.gate2.u_z(blk._ffn(blk.norm2(h1))) - blk.gate2.bg)
        h = blk(h)
        out.append({"entropy": float(ent.mean()), "entropy_per_head": [float(v) for v in ent.mean((0, 2))],
                    "peak_weight": float(w.max(-1).values.mean()), "gate_attn": float(z1.mean()), "gate_ffn": float(z2.mean())})
    delta = mixer.out(h)
    return out, float(delta.norm() / t.norm())


def diagnose(ck_path: str, x: torch.Tensor) -> dict:
    model, a = load(ck_path)
    res = {"checkpoint": ck_path, "trunk": a.trunk, "seed": a.seed}
    with torch.no_grad():
        s, v, p = model.initial_inference(x)
        res["srank_state"], res["srank_state_rel"] = srank(s.flatten(1))
        head_in = F.relu(model.p_bn(model.p_conv(model.pred_res(s)))).flatten(1)
        res["srank_head"], res["srank_head_rel"] = srank(head_in)
    res["dormant"] = dormant_stats(model, x)
    wn = {n: float(sum(q.detach().norm() ** 2 for q in m.parameters()) ** 0.5) for n, m in model.named_children() if list(m.parameters())}
    res["weight_norm"] = float(sum(q.detach().norm() ** 2 for q in model.parameters()) ** 0.5)
    res["weight_norm_modules"] = wn
    if a.trunk == "gtrxl":
        with torch.no_grad():
            s0 = model.repr_res(model.down(x))
            rep, rep_delta = mixer_report(model.repr_mixer, s0)
            act = torch.randint(0, model.action_dim, (x.shape[0],))
            plane = (act.float() / model.action_dim).view(-1, 1, 1, 1).expand(-1, 1, model.hw, model.hw)
            plane = F.relu(model.act_ln(model.act_conv(plane)))
            xd = model.dyn_res(F.relu(model.dyn_bn(model.dyn_conv(torch.cat([s, plane], dim=1))) + s))
            dyn, dyn_delta = mixer_report(model.dyn_mixer, xd)
        res["repr_mixer"], res["repr_mixer_change"] = rep, rep_delta
        res["dyn_mixer"], res["dyn_mixer_change"] = dyn, dyn_delta
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--replay", required=True, help="replay_latest.npz the states are cut from")
    ap.add_argument("--states", type=int, default=512)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    x = stacks_from_replay(a.replay, a.states)
    rows = []
    for ck in a.checkpoints:
        r = diagnose(ck, x)
        rows.append(r)
        d = r["dormant"]
        line = (f"{Path(ck).parents[1].name[:34]:34s} {Path(ck).name[11:23]:12s} srank {r['srank_state']:4d} "
                f"({r['srank_state_rel']:.2f}) head {r['srank_head']:3d}  dormant " +
                " ".join(f"{k[:5]} {100 * v:4.1f}%" for k, v in sorted(d.items())) + f"  |w| {r['weight_norm']:.1f}")
        if "repr_mixer" in r:
            line += (f"  H repr {np.mean([b['entropy'] for b in r['repr_mixer']]):.2f} dyn "
                     f"{np.mean([b['entropy'] for b in r['dyn_mixer']]):.2f}  gate z "
                     f"{np.mean([b['gate_attn'] for b in r['repr_mixer'] + r['dyn_mixer']]):.2f}  "
                     f"mixer dS {r['repr_mixer_change']:.2f}/{r['dyn_mixer_change']:.2f}")
        print(line, flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
