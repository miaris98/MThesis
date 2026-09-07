"""Optimizer and learning-rate-schedule construction for World on Rails training.

Split out of wor_trainer.py because both pieces here encode decisions that were
originally made implicitly against the CNN head, and that turned out to handicap the
Qwen transformer trunk specifically. Keeping them in one place makes them reviewable
as choices rather than as defaults buried in a training loop.
"""
from typing import Dict, List, Tuple
import math

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


#: Parameters that are not weight matrices and should not be weight-decayed. The
#: alpha gates are the important ones: QwenTransformerBlock computes
#: `x + alpha * sublayer(x)` with alpha initialised at 0.1, so decaying alpha toward
#: zero penalises the model for letting each block contribute at all - the exact
#: opposite of what a 12-layer trunk needs in order to route information into its
#: readout token.
NO_DECAY_SUFFIXES = ("alpha_attn", "alpha_ffn", "policy_token", "vision_pos", "type_embed")


def build_param_groups(
    model: nn.Module,
    lr_backbone: float,
    lr_heads: float,
    weight_decay: float,
    decay_gates_and_norms: bool = False
) -> Tuple[List[Dict], Dict[str, int]]:
    """Differential learning rate for backbone vs heads, plus a no-decay group.

    Returns (param_groups, summary). Group order is load-bearing: telemetry reads
    index 0 as the backbone LR and index 1 as the head LR, and the no-decay group at
    index 2 shares the head LR.

    `decay_gates_and_norms=True` reproduces the original behaviour - a single decayed
    group for everything that is not the backbone - so a comparison run can be made
    against it.
    """
    backbone, decay, no_decay = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            backbone.append(param)
        elif decay_gates_and_norms:
            decay.append(param)
        elif (param.ndim <= 1                     # biases, RMSNorm/BatchNorm gains, gates
              or name.endswith(NO_DECAY_SUFFIXES)  # learned tokens and positional tables
              or "embed" in name):                 # nn.Embedding lookup tables
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": decay, "lr": lr_heads, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr_heads, "weight_decay": 0.0}
    ]
    summary = {
        "backbone_params": sum(p.numel() for p in backbone),
        "decayed_params": sum(p.numel() for p in decay),
        "no_decay_params": sum(p.numel() for p in no_decay),
        "no_decay_tensors": len(no_decay)
    }
    return param_groups, summary


def build_optimizer(
    model: nn.Module,
    lr_backbone: float,
    lr_heads: float,
    weight_decay: float,
    decay_gates_and_norms: bool = False,
    verbose: bool = True
) -> AdamW:
    """AdamW over the groups from build_param_groups."""
    groups, summary = build_param_groups(
        model, lr_backbone, lr_heads, weight_decay, decay_gates_and_norms
    )
    if verbose:
        print(f"--> Optimizer groups: backbone {summary['backbone_params']:,} | "
              f"decayed {summary['decayed_params']:,} | "
              f"no-decay {summary['no_decay_params']:,} "
              f"({summary['no_decay_tensors']} tensors: gates, norms, embeddings, learned tokens)")
    return AdamW(groups, weight_decay=weight_decay)


def build_warmup_cosine_scheduler(
    optimizer: AdamW,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_frac: float = 0.05,
    peak_lr: float = 3e-4,
    eta_min: float = 1e-6,
    verbose: bool = True
) -> LambdaLR:
    """Linear warmup into cosine decay, on a per-optimizer-step clock.

    The schedule this replaces was a bare per-epoch CosineAnnealingLR, i.e. the heads
    started at their full LR on batch 1. A conv head tolerates that; a 12-layer
    pre-norm transformer characteristically does not, and the first Qwen run shows the
    signature - 9.2% loss improvement across epochs 2-8 against the CNN's 39.6%, and
    six epochs where the loss rose rather than fell.

    Stepping per batch rather than per epoch matters because warmup over 5% of a
    50-epoch run is 2.5 epochs, which is far too coarse to resolve at epoch
    granularity.
    """
    steps_per_epoch = max(1, steps_per_epoch)
    total_steps = max(1, num_epochs * steps_per_epoch)
    warmup_steps = int(warmup_frac * total_steps)
    # Expressed as a floor ratio so both parameter groups decay to the same relative
    # floor despite starting from different base learning rates.
    min_ratio = eta_min / peak_lr if peak_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    if verbose:
        print(f"--> LR schedule: {warmup_steps} warmup steps "
              f"({warmup_steps / steps_per_epoch:.1f} epochs) then cosine over {total_steps} total steps.")
    return LambdaLR(optimizer, lr_lambda)
