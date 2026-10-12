"""TODO B37 step 1: per-category audit of the GTrXL mixer (how far its tensors moved since init, and the size of their gradients).

The first version divided |W - W0| by |W0| for every tensor. Several mixer tensors start at exactly zero (the zero-initialised output projection `mixer.out`,
which makes the mixer an exact identity at init, and the LayerNorm biases), so their "relative change" was |dW| / 1e-12 = 1e12-1e13 after the first update: a number
that only says "the tensor left zero" and was read as divergence in the 2026-10-05 screens (S-127). `audit` therefore reports

  rel : |W - W0| / max(|W0|, REL_FLOOR)   (bounded for zero-init tensors; read it only for categories that do not start at zero)
  grad: mean gradient norm of the category
  abs : |W - W0| itself (the number to read for zero-init tensors: out, norm biases)

per category, averaged over the tensors of the category.
"""
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

REL_FLOOR = 1e-3


def category(name: str) -> str:
    if any(t in name for t in ("q_proj", "k_proj", "v_proj")):
        return "qkv"
    if "out_proj" in name:
        return "attn_out"
    if any(t in name for t in ("w_gate", "w_up", "w_down")):
        return "ffn"
    if ".gate" in name and name.endswith("weight"):
        return "gate_w"
    if name.endswith(".bg"):
        return "gate_bias"
    if "norm" in name:
        return "norm"
    if "mixer.out" in name or "mixer.inp" in name or name.endswith(".pos"):
        return "io"
    return "other"


def snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Copies of the mixer's parameters (the `W0` of the audit)."""
    return {n: q.detach().clone() for n, q in model.named_parameters() if "mixer" in n}


def audit(model: nn.Module, w0: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """(rel, grad, abs) per category; gradients are those currently stored on the parameters (call before `step()` for the update's own gradients)."""
    rel, grads, ab = {}, {}, {}
    for n, q in model.named_parameters():
        if n not in w0:
            continue
        c = category(n)
        d = (q.detach() - w0[n]).norm()
        rel.setdefault(c, []).append((d / w0[n].norm().clamp_min(REL_FLOOR)).item())
        ab.setdefault(c, []).append(d.item())
        if q.grad is not None:
            grads.setdefault(c, []).append(q.grad.norm().item())
    mean = lambda d: {c: float(np.mean(v)) for c, v in d.items()}  # noqa: E731
    return mean(rel), mean(grads), mean(ab)
