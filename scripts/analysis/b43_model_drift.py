#!/usr/bin/env python3
"""TODO B43 step 0: how far does the learned world model drift from reality as the imagined depth grows past its training horizon?

The dynamics are trained for K = 5 unrolled steps (`--unroll-steps`), but a 64-simulation Gumbel search expands trees up to depth ~17
(S-116), so the search uses the model outside the horizon it was trained on. On real replay trajectories (no search, no policy), roll the
model forward with the *real* actions and compare each imagined step k with what the model itself outputs on the *real* observation at
t + k:

  latent    1 - cosine(projection(imagined state), projection(real state))     (the SimSiam space the consistency loss trains in)
  value     mean |V_imagined - V_real| and the mean signed difference           (V in the h-transformed scale the heads use)
  policy    KL(softmax(policy_real) || softmax(policy_imagined)) and the argmax agreement
  prefix    mean |value prefix predicted - running real reward since the last LSTM reset|   (reset every H = 5 steps as in training)

CPU or one small GPU is enough: a few hundred start states, depth <= 12.

    py scripts/analysis/b43_model_drift.py CKPT REPLAY.npz [--depth 12] [--n 384] [--out drift.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b15_b18_diagnostics import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atari_qwen.models.ez_model import DiscreteSupport  # noqa: E402

H_LSTM = 5  # value-prefix horizon (the LSTM state is reset every H steps, in training and in search)


def stacks(frames, ep_start, t, e, n_stack=4):
    offs = np.arange(-n_stack + 1, 1)
    idx = np.maximum(t[:, None] + offs, ep_start[t, e][:, None])
    f = frames[idx, np.broadcast_to(e[:, None], idx.shape)]            # (n, n_stack, C, H, W)
    return torch.from_numpy(f.reshape(f.shape[0], -1, *f.shape[3:])).float().div_(255.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("replay")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n", type=int, default=384, help="start states")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, args = load(a.checkpoint)
    model = model.to(device).eval()
    support = DiscreteSupport()
    z = np.load(a.replay)
    frames, ep_start, act, rew, done = z["frames"], z["ep_start"], z["act"], z["rew"], z["done"]
    T, E = frames.shape[:2]
    rnd = np.random.default_rng(a.seed)
    D = a.depth
    starts = []
    while len(starts) < a.n:
        t, e = int(rnd.integers(3, T - D - 2)), int(rnd.integers(0, E))
        if not done[t:t + D, e].any():                                    # the real episode continues for D steps
            starts.append((t, e))
    starts = np.array(starts)
    keys = ("latent", "value_abs", "value_signed", "policy_kl", "policy_argmax_agree", "prefix_abs")
    acc = {k: np.zeros((D, 0)) for k in keys}
    cols = {k: [] for k in keys}
    with torch.no_grad():
        for c0 in range(0, len(starts), a.chunk):
            tt, ee = starts[c0:c0 + a.chunk, 0], starts[c0:c0 + a.chunk, 1]
            n = len(tt)
            s, v0, p0 = model.initial_inference(stacks(frames, ep_start, tt, ee).to(device))
            hid = model.init_hidden(n, device)
            run = np.zeros(n, np.float32)
            rows = {k: [] for k in keys}
            for k in range(1, D + 1):
                acts = torch.as_tensor(act[tt + k - 1, ee], device=device)
                s, vp, v, p, hid = model.recurrent_inference(s, acts, hid)
                if (k - 1) % H_LSTM == 0:
                    run = np.zeros(n, np.float32)
                run = run + rew[tt + k - 1, ee]
                sr, vr, pr = model.initial_inference(stacks(frames, ep_start, tt + k, ee).to(device))
                zi, zr = model.project(s, with_grad=False), model.project(sr, with_grad=False)
                vi_s, vr_s = support.vector_to_scalar(v).float(), support.vector_to_scalar(vr).float()
                lp_i, lp_r = F.log_softmax(p.float(), -1), F.log_softmax(pr.float(), -1)
                rows["latent"].append((1 - F.cosine_similarity(zi.float(), zr.float(), dim=-1)).cpu().numpy())
                rows["value_abs"].append((vi_s - vr_s).abs().cpu().numpy())
                rows["value_signed"].append((vi_s - vr_s).cpu().numpy())
                rows["policy_kl"].append((lp_r.exp() * (lp_r - lp_i)).sum(-1).cpu().numpy())
                rows["policy_argmax_agree"].append((p.argmax(-1) == pr.argmax(-1)).float().cpu().numpy())
                pre = support.vector_to_scalar(vp).float().cpu().numpy()
                rows["prefix_abs"].append(np.abs(pre - run))
                if k % H_LSTM == 0:
                    hid = model.init_hidden(n, device)
            for key in keys:
                cols[key].append(np.stack(rows[key]))                    # (D, n)
    res = {k: np.concatenate(v, axis=1).mean(1) for k, v in cols.items()}
    print(f"{Path(a.checkpoint).parents[1].name[:34]} {Path(a.checkpoint).name[11:30]}  trunk {args.trunk}  starts {len(starts)}  "
          f"(K = {getattr(args, 'unroll_steps', 5)}, LSTM reset every {H_LSTM})")
    print(f"{'depth':>5s} {'latent 1-cos':>13s} {'|dV|':>8s} {'dV signed':>10s} {'policy KL':>10s} {'argmax ok':>10s} {'|prefix err|':>13s}")
    for k in range(D):
        print(f"{k + 1:5d} {res['latent'][k]:13.4f} {res['value_abs'][k]:8.3f} {res['value_signed'][k]:+10.3f} {res['policy_kl'][k]:10.4f} "
              f"{res['policy_argmax_agree'][k]:10.3f} {res['prefix_abs'][k]:13.4f}")
    inh, out = slice(0, 5), slice(5, D)
    print("mean depth 1-5 vs 6-%d:  " % D + "  ".join(f"{k} {res[k][inh].mean():.4f} -> {res[k][out].mean():.4f}" for k in
                                                       ("latent", "value_abs", "policy_kl", "prefix_abs")))
    if a.out:
        json.dump({k: v.tolist() for k, v in res.items()}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
