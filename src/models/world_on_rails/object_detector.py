"""Object detector on the frozen RGB features (2026-11-18, step 2 of the teacher ladder: the thesis-legal object-token student).

The privileged teacher drives from up to 20 object tokens (build_box_targets.box_token: x/20, y/10, cos yaw, sin yaw, ex/3, ey/3, speed/10,
red, yellow, 6-class one-hot, valid). This head predicts the same tokens from the frozen encoder's full stride-32 map (PCA-256, 9x24 at the
288x768 geometry), so a teacher-style trunk can drive from objects while the only input at inference stays the single RGB camera (user
ruling 2026-11-18). Logged boxes are training labels only.

DETR-style: Q learned queries cross-attend over the map cells (+ sine position code); each query outputs a box (x, y, cos, sin, ex, ey in
token units), class logits, traffic-light state logits and an object/no-object logit. Training matches queries to the ground-truth objects
with the Hungarian algorithm (scipy; a greedy fallback if scipy is missing). Speed is not predicted: one frame does not show it (targets
are the `_visns` tokens).
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.training.build_box_targets import BOX_CLASSES, BOX_FEATS, VALID_SLOT

N_CLS = len(BOX_CLASSES)        # 6
REG = slice(0, 6)               # x/20, y/10, cos, sin, ex/3, ey/3
CLS = slice(9, 9 + N_CLS)       # one-hot in the token
LIGHT_CLS = BOX_CLASSES.index("traffic_light")


def sine_pos(h: int, w: int, dim: int, device) -> torch.Tensor:
    """(h*w, dim) 2-D sine/cosine position code (rows then columns)."""
    q = dim // 4
    freq = 1.0 / (10000 ** (torch.arange(q, dtype=torch.float32, device=device) / q))
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device), torch.arange(w, dtype=torch.float32, device=device), indexing="ij")
    ay, ax = ys.reshape(-1, 1) * freq, xs.reshape(-1, 1) * freq
    pos = torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], dim=1)
    if pos.shape[1] < dim:
        pos = torch.cat([pos, torch.zeros(pos.shape[0], dim - pos.shape[1], device=device)], dim=1)
    return pos


def pca_project(fmap: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) full map -> (B, K, H, W) PCA coordinates exactly as build_pooled_cache.py --pca_full writes them (basis (K, C);
    the cache stores float16, so the result is rounded through float16)."""
    return torch.einsum("bchw,kc->bkhw", fmap.float() - mean.float().view(1, -1, 1, 1), basis.float()).half().float()


class ObjectDetector(nn.Module):
    def __init__(self, in_ch: int = 256, dim: int = 256, queries: int = 30, layers: int = 3, heads: int = 8):
        super().__init__()
        self.in_ch, self.dim, self.queries = in_ch, dim, queries
        self.cell = nn.Sequential(nn.Linear(in_ch, dim), nn.LayerNorm(dim))
        self.query = nn.Parameter(torch.randn(1, queries, dim) * 0.02)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim_feedforward=4 * dim, dropout=0.0, batch_first=True, norm_first=True),
            num_layers=layers, norm=nn.LayerNorm(dim))
        self.box = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 6))
        self.cls = nn.Linear(dim, N_CLS)
        self.light = nn.Linear(dim, 3)      # none / red / yellow (green and unknown = none, as in box_token)
        self.obj = nn.Linear(dim, 1)
        self._pos: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def forward(self, fmap: torch.Tensor) -> Dict[str, torch.Tensor]:
        """fmap (B, C, H, W) -> per query: box (B, Q, 6), cls (B, Q, 6), light (B, Q, 3), obj (B, Q) logits."""
        B, C, H, W = fmap.shape
        key = (H, W, str(fmap.device))
        if key not in self._pos:
            self._pos[key] = sine_pos(H, W, self.dim, fmap.device).unsqueeze(0)
        mem = self.cell(fmap.flatten(2).transpose(1, 2).float()) + self._pos[key]
        h = self.dec(self.query.expand(B, -1, -1), mem)
        return {"box": self.box(h), "cls": self.cls(h), "light": self.light(h), "obj": self.obj(h).squeeze(-1)}

    @torch.no_grad()
    def tokens(self, out: Dict[str, torch.Tensor], m: int = 20, threshold: float = 0.5, max_range: float = 0.0) -> torch.Tensor:
        """Teacher-format tokens (B, m, BOX_FEATS) from the predictions: objects with p(obj) >= threshold, nearest first, speed 0.
        max_range > 0 also drops objects farther than max_range metres (R1, 2026-11-19: recall is 0.64-0.82 within 20 m, 0.26 at 30-50 m)."""
        B, Q = out["obj"].shape
        p = out["obj"].sigmoid()
        box = out["box"].float()
        dist = torch.hypot(box[..., 0] * 20.0, box[..., 1] * 10.0)
        ok = p >= threshold
        if max_range > 0:
            ok = ok & (dist <= max_range)
        dist = torch.where(ok, dist, torch.full_like(dist, 1e9))
        k = min(m, Q)
        order = dist.argsort(dim=1)[:, :k]
        tok = torch.zeros(B, m, BOX_FEATS, device=box.device)
        b = torch.arange(B, device=box.device).unsqueeze(1)
        tok[:, :k, 0:6] = box[b, order]
        lt = out["light"][b, order].argmax(-1)
        tok[:, :k, 7] = (lt == 1).float()
        tok[:, :k, 8] = (lt == 2).float()
        tok[:, :k, 9:9 + N_CLS] = F.one_hot(out["cls"][b, order].argmax(-1), N_CLS).float()
        tok[:, :k, VALID_SLOT] = 1.0
        keep = torch.zeros(B, m, device=box.device)
        keep[:, :k] = (dist[b, order] < 1e8).float()
        return tok * keep.unsqueeze(-1)


def _assign(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except ImportError:   # greedy fallback: cheapest pair first
        rows, cols, used_r, used_c = [], [], set(), set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), cost.shape[1])
            if r not in used_r and c not in used_c:
                rows.append(r); cols.append(c); used_r.add(r); used_c.add(c)
            if len(rows) == min(cost.shape):
                break
        return np.array(rows), np.array(cols)


def match(out: Dict[str, torch.Tensor], gt: torch.Tensor) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Hungarian matching per sample: (query indices, gt indices). Cost = L1 of the box in metres (x, y) and token units (rest) + class.
    The (B, Q, M) cost is built in one batched pass on the model's device; only the small assignment runs per sample on the CPU."""
    with torch.no_grad():
        box = out["box"].float()                                   # (B, Q, 6)
        logp = out["cls"].float().log_softmax(-1)                  # (B, Q, C)
        gt = gt.float().to(box.device)                             # (B, M, F)
        valid = (gt[..., VALID_SLOT] > 0.5).cpu().numpy()          # (B, M)
        dx = (box[:, :, None, 0] - gt[:, None, :, 0]).abs() * 20.0
        dy = (box[:, :, None, 1] - gt[:, None, :, 1]).abs() * 10.0
        drest = (box[:, :, None, 2:6] - gt[:, None, :, 2:6]).abs().sum(-1)
        gcls = gt[..., CLS].argmax(-1)                             # (B, M)
        ccls = -torch.gather(logp, 2, gcls[:, None, :].expand(-1, box.shape[1], -1))
        cost = (0.2 * (dx + dy) + drest + ccls).cpu().numpy()      # (B, Q, M)
        res = []
        for i in range(cost.shape[0]):
            cols = np.nonzero(valid[i])[0]
            if len(cols) == 0:
                res.append((np.zeros(0, np.int64), np.zeros(0, np.int64)))
                continue
            q, j = _assign(cost[i][:, cols])
            res.append((q.astype(np.int64), cols[j].astype(np.int64)))
        return res


def detector_loss(out: Dict[str, torch.Tensor], gt: torch.Tensor, w_obj: float = 1.0, w_box: float = 5.0, w_cls: float = 1.0,
                  w_light: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    pairs = match(out, gt)
    B, Q = out["obj"].shape
    dev = out["obj"].device
    obj_t = torch.zeros(B, Q, device=dev)
    bi, qi, gi = [], [], []
    for i, (q, j) in enumerate(pairs):
        obj_t[i, torch.as_tensor(q, device=dev)] = 1.0
        bi += [i] * len(q); qi += list(q); gi += list(j)
    # gt rows of the matched objects (gt valid rows are packed first by build_box_targets: nearest first, padding after)
    l_obj = F.binary_cross_entropy_with_logits(out["obj"].float(), obj_t)
    if bi:
        bi_t, qi_t, gi_t = (torch.as_tensor(v, device=dev) for v in (bi, qi, gi))
        g = gt.float()[bi_t, gi_t]
        l_box = F.l1_loss(out["box"].float()[bi_t, qi_t], g[:, REG])
        l_cls = F.cross_entropy(out["cls"].float()[bi_t, qi_t], g[:, CLS].argmax(-1))
        lt = (g[:, 7] > 0.5).long() + 2 * (g[:, 8] > 0.5).long()
        is_light = g[:, CLS].argmax(-1) == LIGHT_CLS
        l_light = F.cross_entropy(out["light"].float()[bi_t, qi_t][is_light], lt[is_light]) if bool(is_light.any()) else out["obj"].sum() * 0
    else:
        l_box = l_cls = l_light = out["obj"].sum() * 0
    loss = w_obj * l_obj + w_box * l_box + w_cls * l_cls + w_light * l_light
    return loss, {"obj": float(l_obj.detach()), "box": float(l_box.detach()), "cls": float(l_cls.detach()), "light": float(l_light.detach())}


DIST_BINS = ((0, 10), (10, 20), (20, 30), (30, 50))


class DetectionMeter:
    """Recall / precision at a centre-distance tolerance (metres), position error of matched objects, by distance bin and class."""

    def __init__(self, tol_m: float = 2.0, threshold: float = 0.5):
        self.tol, self.thr = tol_m, threshold
        self.gt = np.zeros((len(DIST_BINS), N_CLS)); self.hit = np.zeros_like(self.gt); self.err = np.zeros_like(self.gt)
        self.pred = 0; self.tp = 0; self.cls_ok = 0

    def update(self, out: Dict[str, torch.Tensor], gt: torch.Tensor) -> None:
        p = out["obj"].detach().sigmoid().float().cpu().numpy()
        box = out["box"].detach().float().cpu().numpy()
        cls = out["cls"].detach().argmax(-1).cpu().numpy()
        gt = gt.detach().float().cpu().numpy()
        for i in range(gt.shape[0]):
            keep = p[i] >= self.thr
            pxy = np.stack([box[i, keep, 0] * 20.0, box[i, keep, 1] * 10.0], 1)
            pc = cls[i, keep]
            g = gt[i][gt[i, :, VALID_SLOT] > 0.5]
            gxy = np.stack([g[:, 0] * 20.0, g[:, 1] * 10.0], 1) if len(g) else np.zeros((0, 2))
            gc = g[:, CLS].argmax(-1) if len(g) else np.zeros(0, int)
            self.pred += len(pxy)
            used = set()
            if len(g) and len(pxy):
                d = np.linalg.norm(gxy[:, None] - pxy[None], axis=2)
                for gi in np.argsort(d.min(1)):   # each ground-truth object takes its nearest unused prediction within tol
                    cand = [j for j in np.argsort(d[gi]) if j not in used and d[gi, j] <= self.tol]
                    rng = float(np.hypot(*gxy[gi]))
                    b = next((k for k, (a, c) in enumerate(DIST_BINS) if a <= rng < c), len(DIST_BINS) - 1)
                    self.gt[b, gc[gi]] += 1
                    if cand:
                        j = cand[0]; used.add(j)
                        self.hit[b, gc[gi]] += 1; self.err[b, gc[gi]] += d[gi, j]; self.tp += 1; self.cls_ok += int(pc[j] == gc[gi])
            else:
                for gi in range(len(g)):
                    rng = float(np.hypot(*gxy[gi]))
                    b = next((k for k, (a, c) in enumerate(DIST_BINS) if a <= rng < c), len(DIST_BINS) - 1)
                    self.gt[b, gc[gi]] += 1

    def summary(self) -> Dict[str, float]:
        s = {"recall": self.hit.sum() / max(self.gt.sum(), 1), "precision": self.tp / max(self.pred, 1),
             "pos_err_m": self.err.sum() / max(self.hit.sum(), 1), "cls_acc": self.cls_ok / max(self.tp, 1), "n_gt": float(self.gt.sum())}
        for k, (a, c) in enumerate(DIST_BINS):
            s[f"recall_{a}_{c}m"] = self.hit[k].sum() / max(self.gt[k].sum(), 1)
            s[f"pos_err_{a}_{c}m"] = self.err[k].sum() / max(self.hit[k].sum(), 1)
        for ci, name in enumerate(BOX_CLASSES):
            s[f"recall_{name}"] = self.hit[:, ci].sum() / max(self.gt[:, ci].sum(), 1)
        return s
