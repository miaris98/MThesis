"""TODO A84: held-out readouts on the reason labels, outside val_loss (which stays the waypoint L1 every earlier run is stated in).

For ANY checkpoint validated on a labelled data dir (no reason heads needed) - what the waypoint L1 cannot see:
  val_reason_speed_err_<cause>   mean (predicted - expert) target speed in m/s on frames whose speed is bound by that cause (+ = too fast)
  val_reason_pstop_danger        mean P(stop bin) on danger frames (the expert must not go on)
  val_reason_speed_err_danger    mean speed error on danger frames
With the heads:
  val_reason_cause_acc / _macro_recall, val_reason_dist_mae_m, val_reason_danger_auroc / _ap / _rate, val_reason_brake_auroc
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

from src.training.reason_labels import REASON_CLASSES


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUROC with average ranks for ties; NaN when one class is missing."""
    pos, neg = int(labels.sum()), int((1 - labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / hits.sum())


class ReasonValAccumulator:
    def __init__(self, speed_head=None):
        self.speed_head = speed_head
        k = len(REASON_CLASSES)
        self.err_sum, self.err_n = np.zeros(k), np.zeros(k)
        self.cm = np.zeros((k, k))
        self.dist_abs, self.dist_n = 0.0, 0
        self.dg_err, self.dg_pstop, self.dg_n = 0.0, 0.0, 0
        self.scores: Dict[str, List[np.ndarray]] = {"danger": [], "brake": []}
        self.labels: Dict[str, List[np.ndarray]] = {"danger": [], "brake": []}

    @torch.no_grad()
    def add(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], device) -> None:
        if "reason_cause" not in batch:
            return
        cause = batch["reason_cause"].to(device).long()
        danger = batch["reason_danger"].to(device).float()
        if self.speed_head is not None and "target_speed_logits" in out and "target_speed" in batch:
            logits = out["target_speed_logits"].float()
            err = (self.speed_head.expected_speed(logits).float() - batch["target_speed"].to(device).float()).cpu().numpy()
            c = cause.cpu().numpy()
            for k in range(len(REASON_CLASSES)):
                m = c == k
                self.err_sum[k] += float(err[m].sum())
                self.err_n[k] += int(m.sum())
            dm = (danger == 1).cpu().numpy()
            self.dg_err += float(err[dm].sum())
            self.dg_pstop += float(logits.softmax(-1)[:, 0].cpu().numpy()[dm].sum())
            self.dg_n += int(dm.sum())
        if "reason_cause_logits" in out:
            v = cause >= 0
            pred = out["reason_cause_logits"].argmax(-1)
            np.add.at(self.cm, (cause[v].cpu().numpy(), pred[v].cpu().numpy()), 1)
            dv = batch["reason_dist_valid"].to(device).float() > 0
            if bool(dv.any()):
                d_pred = torch.expm1(out["reason_dist"].float()[dv].clamp(max=8.0))
                d_true = torch.expm1(batch["reason_dist"].to(device).float()[dv])
                self.dist_abs += float((d_pred - d_true).abs().sum())
                self.dist_n += int(dv.sum())
            for key in ("danger", "brake"):
                y = batch[f"reason_{key}"].to(device).float()
                m = y >= 0
                if bool(m.any()):
                    self.scores[key].append(torch.sigmoid(out[f"reason_{key}_logit"].float()[m]).cpu().numpy())
                    self.labels[key].append(y[m].cpu().numpy())

    def result(self) -> Dict[str, float]:
        r: Dict[str, float] = {}
        for k, name in enumerate(REASON_CLASSES):
            if self.err_n[k] > 0:
                r[f"val_reason_speed_err_{name}"] = self.err_sum[k] / self.err_n[k]
                r[f"val_reason_n_{name}"] = float(self.err_n[k])
        if self.dg_n:
            r.update(val_reason_speed_err_danger=self.dg_err / self.dg_n, val_reason_pstop_danger=self.dg_pstop / self.dg_n,
                     val_reason_n_danger=float(self.dg_n))
        tot = self.cm.sum()
        if tot > 0:
            r["val_reason_cause_acc"] = float(np.trace(self.cm) / tot)
            rec = [self.cm[k, k] / self.cm[k].sum() for k in range(len(REASON_CLASSES)) if self.cm[k].sum() > 0]
            r["val_reason_cause_macro_recall"] = float(np.mean(rec))
        if self.dist_n:
            r["val_reason_dist_mae_m"] = self.dist_abs / self.dist_n
        for key in ("danger", "brake"):
            if self.scores[key]:
                s, y = np.concatenate(self.scores[key]), np.concatenate(self.labels[key])
                r[f"val_reason_{key}_auroc"] = auroc(s, y)
                r[f"val_reason_{key}_ap"] = average_precision(s, y)
                r[f"val_reason_{key}_rate"] = float(y.mean())
        # Plain Python floats: these land in the checkpoint's "metrics", and torch >= 2.6 loads checkpoints weights_only, which refuses numpy
        # scalars - a numpy float here made every A84 checkpoint unloadable at evaluation (caught in the 2026-11-16 smoke run).
        return {k: float(v) for k, v in r.items() if not math.isnan(float(v))}
