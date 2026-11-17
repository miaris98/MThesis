"""Read-only probe of what the qwen policy looked at and decided, per closed-loop tick (2026-11-12, attention-map videos).

The trunk's attention runs through F.scaled_dot_product_attention, which returns no weights. A forward hook on every block's
attention module recomputes the policy token's attention row from the module's own input:

    softmax(q_policy . k_all / sqrt(head_dim))     (with the block's QK-norm when the checkpoint has one)

which is the same quantity SDPA computes internally (same scale, no mask, no dropout at eval). The hook returns nothing, so the
module's output - and therefore every control - is unchanged.

Token order in the trunk (QwenWaypointTransformer.forward): [policy, vision x N, speed, route, command], so for a 4x4 grid the
policy token's row has 1 + 16 + 3 = 20 entries; entries 1..16 are the vision cells in row-major order (top-left first).

`decision()` reads the speed head: the posterior over the speed bins, and for checkpoints with speed-head experts (TODO A81) the
gate's probabilities, the top-k experts with their weights, and the expected target speed with and without the experts' correction.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class PolicyAttentionProbe:
    def __init__(self, net):
        trunk = getattr(net, "trunk", None)
        if trunk is None or not hasattr(trunk, "blocks"):
            raise RuntimeError(f"{type(net).__name__} has no qwen trunk to probe")
        self.net = net
        self.n_layers = len(trunk.blocks)
        self.latest: List[Optional[np.ndarray]] = [None] * self.n_layers   # per layer: (heads, tokens) attention of the policy token
        self.handles = [blk.attn.register_forward_hook(self._hook(i)) for i, blk in enumerate(trunk.blocks)]

    def _hook(self, i: int):
        def fn(mod, inputs, output):
            x = inputs[0]
            with torch.no_grad():
                B, N, _ = x.shape
                q = mod.q_proj(x[:1, :1]).view(1, 1, mod.num_heads, mod.head_dim).transpose(1, 2)    # policy token of sample 0
                k = mod.k_proj(x[:1]).view(1, N, mod.num_heads, mod.head_dim).transpose(1, 2)
                if getattr(mod, "q_norm", None) is not None:
                    q, k = mod.q_norm(q), mod.k_norm(k)
                w = (q.float() @ k.float().transpose(-2, -1) * mod.scale).softmax(-1)              # (1, heads, 1, N)
                self.latest[i] = w[0, :, 0].cpu().numpy()
        return fn

    def layout(self) -> Dict:
        net = self.net
        gh, gw = getattr(net, "vision_grid_hw", (0, 0))
        n_vis = int(getattr(net.trunk, "num_vision_tokens", gh * gw))
        return {"n_layers": self.n_layers, "grid_hw": [int(gh), int(gw)], "n_vision": n_vis,
                "tokens": ["policy"] + [f"cell{i}" for i in range(n_vis)] + ["speed", "route", "command"]}

    def attention(self) -> Optional[List[List[float]]]:
        """(layers, tokens) head-averaged policy-token attention of the latest forward; None before the first forward."""
        if any(a is None for a in self.latest):
            return None
        return [np.round(a.mean(0), 4).tolist() for a in self.latest]

    @torch.no_grad()
    def decision(self, speed_mps: float) -> Dict:
        """Speed head and (if present) speed-head experts for the latest forward. Reads the token act() just used; changes nothing."""
        net, out = self.net, {}
        head = getattr(net, "target_speed_head", None)
        pr = getattr(net, "_last_policy_repr", None)
        if head is None or pr is None:
            return out
        out["bins_kmh"] = [round(float(b) * 3.6, 2) for b in head.bins.detach().cpu().reshape(-1)]
        moe = getattr(net, "speed_moe", None)
        if moe is not None and not getattr(net, "path_cascade", False) and getattr(net, "target_speed_input", "policy") == "policy":
            from src.models.world_on_rails.qwen_wor_policy import speed_moe_inputs
            x = speed_moe_inputs(pr.float(), torch.tensor([speed_mps], device=pr.device))
            g = moe.gate(x.float()).float().softmax(-1)[0]
            top_p, top_i = g.topk(moe.top_k)
            base = head(pr)
            delta = moe(x).to(base.dtype)
            out["moe"] = {
                "gate": np.round(g.cpu().numpy(), 3).tolist(),
                "top": [int(i) for i in top_i.cpu()],
                "top_w": [round(float(w), 3) for w in (top_p / top_p.sum()).cpu()],
                "standing": bool(speed_mps < 0.5),
                "tgt_base_kmh": round(3.6 * float(head.expected_speed(base)[0]), 2),
                "tgt_moe_kmh": round(3.6 * float(head.expected_speed(base + delta)[0]), 2),
            }
        return out

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
