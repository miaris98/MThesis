"""Batched Gumbel MuZero search (EfficientZero V2 Atari variant), fully tensorised on the GPU.

Semantics follow EZ-V2's reference search (ez/mcts/py_mcts.py + cy_mcts.py), which is what their
Breakout numbers were produced with:
  * root: Gumbel-top-m sampling (g + logits), sequential halving, equal visits among the survivors;
    the gumbel noise is NOT scaled by temperature (commented out in their cython path);
  * non-root: argmax(improved_policy - N(a) / (1 + sum N));
  * improved policy = softmax(logits + sigma(completed Q)), sigma(q) = (c_visit + max N) * c_scale * q,
    completed Q = Q(s,a) for visited children and v_mix otherwise, min-max normalised per tree
    (soft delta 0.01, clipped to [0, 1]);
  * rewards come from the value-prefix LSTM: r = prefix(child) - prefix(parent), with the prefix and
    LSTM state reset every `lstm_horizon` depths.
Instead of Python node objects (EZ-V2 needed cython for speed), every tree is a row of fixed-size
tensors (num_simulations + 1 nodes), so one search over B roots costs num_simulations batched model
calls plus a few small kernels per tree level.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def halving_schedule(num_simulations: int, num_top_actions: int):
    """{simulation index after which to halve: new m}, reproducing EZ-V2's ready_for_next_gumble_phase."""
    n, m = num_simulations, num_top_actions
    cur_m, used = m, 0
    nxt = max(np.floor(n / (np.log2(m) * cur_m)), 1) * cur_m if m > 1 else n
    sched = {}
    for sim in range(n):
        if (sim + 1) >= nxt and cur_m > 1:
            cur_m //= 2
            if cur_m > 2:
                extra = max(np.floor(n / (np.log2(m) * cur_m)), 1) * cur_m
            else:
                extra = n - used
            used += extra
            nxt = min(nxt + extra, n)
            sched[sim] = max(cur_m, 1)
    return sched


class GumbelMCTS:
    def __init__(self, num_actions: int, support, num_simulations: int = 16, num_top_actions: int = 4,
                 discount: float = 0.997 ** 4, c_visit: float = 50.0, c_scale: float = 0.1,
                 minmax_delta: float = 0.01, lstm_horizon: int = 5):
        self.A = num_actions
        self.S = num_simulations
        self.m = min(num_top_actions, num_actions)
        self.gamma = discount
        self.c_visit, self.c_scale, self.delta = c_visit, c_scale, minmax_delta
        self.H = lstm_horizon
        self.support = support
        self.schedule = halving_schedule(num_simulations, self.m)

    # --------------------------------------------------------------------------------------------
    def _normalize(self, q, mn, mx):
        ok = (mx > mn).unsqueeze(-1)
        qc = torch.minimum(torch.maximum(q, mn.unsqueeze(-1)), mx.unsqueeze(-1))
        qn = (qc - mn.unsqueeze(-1)) / torch.clamp(mx - mn, min=self.delta).unsqueeze(-1)
        return torch.where(ok, qn, q).clamp(0.0, 1.0)

    def _transformed_completed_q(self, nodes):
        """sigma(completed Q) for node index `nodes` (B,) of every tree -> (B, A)."""
        b = self.ar
        ch = self.child[b, nodes]                                    # (B, A)
        exp = ch >= 0
        chc = ch.clamp(min=0)
        c_vis = torch.where(exp, self.visit[b.unsqueeze(1), chc], torch.zeros_like(chc, dtype=torch.float32))
        c_val = self.vsum[b.unsqueeze(1), chc] / self.visit[b.unsqueeze(1), chc].clamp(min=1)
        q = self.reward[b.unsqueeze(1), chc] + self.gamma * c_val
        # v_mix (Gumbel MuZero appendix D)
        pi = torch.softmax(self.logits[b, nodes], dim=-1)
        pi_sum = (pi * exp).sum(-1)
        pi_q = (pi * q * exp).sum(-1)
        v_node = self.vsum[b, nodes] / self.visit[b, nodes].clamp(min=1)
        vis_sum = c_vis.sum(-1)
        v_mix = torch.where(pi_sum < 1e-6, v_node,
                            (v_node + vis_sum * pi_q / pi_sum.clamp(min=1e-6)) / (1 + vis_sum))
        completed = torch.where(exp, q, v_mix.unsqueeze(-1))
        qn = self._normalize(completed, self.mn, self.mx)
        return (self.c_visit + c_vis.max(-1).values).unsqueeze(-1) * self.c_scale * qn, c_vis

    # --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def search(self, model, root_states: torch.Tensor, root_values: torch.Tensor, root_logits: torch.Tensor,
               add_noise: bool = True, g: torch.Tensor = None, depth_bound: int = None):
        """-> (root search values (B,), improved policies (B, A), best actions (B,)) as numpy.

        g: the Gumbel noise (B, A) to use instead of sampling it (add_noise is then ignored); zeros = no noise.
        depth_bound: sync-free mode for CUDA-graph capture (`GraphedSearch`). Selection runs exactly `depth_bound` tree
        levels and the backup `depth_bound + 1`, instead of looping until `active.any()` / `alive.any()` (a GPU-to-host
        sync per level). Extra iterations change nothing (finished rows are masked), so the result is bit-identical to
        the default mode unless a tree is deeper than the bound; the returned `overflow` (a device bool) is then True and
        the caller must redo the search without the bound. Returns device tensors (values, policies, best, overflow)."""
        dev = root_states.device
        B, A, N = root_states.shape[0], self.A, self.S + 1
        self.ar = torch.arange(B, device=dev)
        f32 = dict(device=dev, dtype=torch.float32)
        self.visit = torch.zeros(B, N, **f32)
        self.vsum = torch.zeros(B, N, **f32)
        self.reward = torch.zeros(B, N, **f32)
        self.prefix = torch.zeros(B, N, **f32)
        self.logits = torch.zeros(B, N, A, **f32)
        self.child = torch.full((B, N, A), -1, device=dev, dtype=torch.long)
        self.parent = torch.full((B, N), -1, device=dev, dtype=torch.long)
        self.depth = torch.zeros(B, N, device=dev, dtype=torch.long)
        self.mn = torch.full((B,), float("inf"), **f32)
        self.mx = torch.full((B,), -float("inf"), **f32)
        states = torch.zeros((N,) + tuple(root_states.shape), device=dev, dtype=root_states.dtype)
        states[0] = root_states
        hid = model.init_hidden(B, dev)
        hid_h = torch.zeros((N,) + tuple(hid[0].shape[1:]), device=dev, dtype=hid[0].dtype)
        hid_c = torch.zeros_like(hid_h)

        self.visit[:, 0] = 1.0
        self.vsum[:, 0] = root_values.float()
        self.logits[:, 0] = root_logits.float()
        if g is None:
            g = (torch.distributions.Gumbel(0.0, 1.0).sample((B, A)).to(dev) if add_noise
                 else torch.zeros(B, A, device=dev))
        fixed = depth_bound is not None
        overflow = torch.zeros((), device=dev, dtype=torch.bool)
        sel = torch.argsort(g + self.logits[:, 0], dim=-1, descending=True)   # root candidates, best first
        m_cur = self.m
        b = self.ar

        for sim in range(self.S):
            # ---- selection -------------------------------------------------------------------
            cur = torch.zeros(B, device=dev, dtype=torch.long)
            active = torch.ones(B, device=dev, dtype=torch.bool)
            leaf_parent = torch.zeros_like(cur)
            leaf_action = torch.zeros_like(cur)
            for _ in range(depth_bound if fixed else self.S + 1):
                # root rule: first of the top-m candidates with the fewest visits
                cand = sel[:, :m_cur]
                cand_ch = self.child[b.unsqueeze(1), 0, cand]
                cand_vis = torch.where(cand_ch >= 0, self.visit[b.unsqueeze(1), cand_ch.clamp(min=0)],
                                       torch.zeros_like(cand, dtype=torch.float32))
                a_root = cand[b, torch.argmin(cand_vis, dim=-1)]              # argmin = first minimum
                # non-root rule
                tq, c_vis = self._transformed_completed_q(cur)
                improved = torch.softmax(self.logits[b, cur] + tq, dim=-1)
                score = improved - c_vis / (1 + c_vis.sum(-1, keepdim=True))
                a_non = score.argmax(-1)
                a = torch.where(cur == 0, a_root, a_non)
                nxt = self.child[b, cur, a]
                is_leaf = active & (nxt < 0)
                leaf_parent = torch.where(is_leaf, cur, leaf_parent)
                leaf_action = torch.where(is_leaf, a, leaf_action)
                active = active & ~is_leaf
                cur = torch.where(active, nxt, cur)
                if not fixed and not bool(active.any()):
                    break
            if fixed:
                overflow = overflow | active.any()

            # ---- expansion ---------------------------------------------------------------------
            new = sim + 1
            p_states = states[leaf_parent, b]
            h_in = (hid_h[leaf_parent, b].unsqueeze(0).contiguous(), hid_c[leaf_parent, b].unsqueeze(0).contiguous())
            s2, vp_logits, v_logits, p_logits, h_out = model.recurrent_inference(p_states, leaf_action, h_in)
            vp = self.support.vector_to_scalar(vp_logits)
            v = self.support.vector_to_scalar(v_logits)
            states[new] = s2.to(states.dtype)
            d_new = self.depth[b, leaf_parent] + 1
            reset_here = (d_new % self.H) == 0
            hid_h[new] = torch.where(reset_here.unsqueeze(-1), torch.zeros_like(h_out[0][0]), h_out[0][0])
            hid_c[new] = torch.where(reset_here.unsqueeze(-1), torch.zeros_like(h_out[1][0]), h_out[1][0])
            self.child[b, leaf_parent, leaf_action] = torch.full_like(leaf_action, new)  # a tensor: a Python int is a CPU copy
            self.parent[:, new] = leaf_parent
            self.depth[:, new] = d_new
            self.logits[:, new] = p_logits.float()
            self.prefix[:, new] = vp
            parent_reset = (self.depth[b, leaf_parent] % self.H) == 0
            self.reward[:, new] = vp - torch.where(parent_reset, torch.zeros_like(vp), self.prefix[b, leaf_parent])

            # ---- backup ------------------------------------------------------------------------
            node = torch.full((B,), new, device=dev, dtype=torch.long)
            value = v
            alive = torch.ones(B, device=dev, dtype=torch.bool)
            levels = 0
            while (levels <= depth_bound) if fixed else bool(alive.any()):  # fixed: bounded, no sync; default: until done
                levels += 1
                nd = node.clamp(min=0)
                self.vsum[b, nd] += torch.where(alive, value, torch.zeros_like(value))
                self.visit[b, nd] += alive.float()
                value = self.reward[b, nd] + self.gamma * value
                self.mn = torch.where(alive, torch.minimum(self.mn, value), self.mn)
                self.mx = torch.where(alive, torch.maximum(self.mx, value), self.mx)
                node = torch.where(alive, self.parent[b, nd], node)
                alive = alive & (node >= 0)
            if fixed:
                overflow = overflow | alive.any()

            # ---- sequential halving -----------------------------------------------------------
            if sim in self.schedule:
                new_m = self.schedule[sim]
                tq_root, _ = self._transformed_completed_q(torch.zeros(B, device=dev, dtype=torch.long))
                cand = sel[:, :m_cur]
                sc = (g + self.logits[:, 0] + tq_root).gather(1, cand)
                order = torch.argsort(sc, dim=-1, descending=True)
                sel = cand.gather(1, order)[:, :new_m]
                m_cur = new_m

        tq_root, _ = self._transformed_completed_q(torch.zeros(B, device=dev, dtype=torch.long))
        policy = torch.softmax(self.logits[:, 0] + tq_root, dim=-1)
        root_value = self.vsum[:, 0] / self.visit[:, 0]
        best = sel[:, 0]
        if fixed:
            return root_value, policy, best, overflow
        return root_value.cpu().numpy(), policy.cpu().numpy(), best.cpu().numpy()


class GraphedSearch:
    """`GumbelMCTS.search` replayed from a CUDA graph (TODO B23 step 1).

    One 16-simulation search is ~7,400 tiny kernel launches and ~250 GPU-to-host syncs for ~17 ms of GPU work, so an
    evaluation process is bound by one CPU core, not by the GPU. The search is captured once per batch size with a fixed
    tree depth bound (see `search(depth_bound=...)`) and replayed with one launch. Results are bit-identical to the
    eager search; a tree deeper than the bound (counted in `fallbacks`) is re-searched eagerly with the same noise."""

    def __init__(self, mcts: "GumbelMCTS", model, depth_bound: int = 6):
        self.mcts, self.model, self.depth = mcts, model, depth_bound
        self.entries, self.fallbacks, self.calls = {}, 0, 0

    def _capture(self, states, values, logits):
        B, A = states.shape[0], self.mcts.A
        e = {"s": states.clone(), "v": values.clone().float(), "l": logits.clone().float(),
             "g": torch.zeros(B, A, device=states.device)}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warm-up: cuDNN / cuBLAS handles and allocator pools before capture
            for _ in range(2):
                self.mcts.search(self.model, e["s"], e["v"], e["l"], g=e["g"], depth_bound=self.depth)
        torch.cuda.current_stream().wait_stream(stream)
        e["graph"] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(e["graph"]):
            e["out"] = self.mcts.search(self.model, e["s"], e["v"], e["l"], g=e["g"], depth_bound=self.depth)
        return e

    @torch.no_grad()
    def __call__(self, root_states, root_values, root_logits, add_noise: bool = True):
        self.calls += 1
        key = (tuple(root_states.shape), root_states.dtype, self.mcts.S)
        if key not in self.entries:
            self.entries[key] = self._capture(root_states, root_values, root_logits)
        e = self.entries[key]
        B, A = root_states.shape[0], self.mcts.A
        g = (torch.distributions.Gumbel(0.0, 1.0).sample((B, A)).to(root_states.device) if add_noise
             else torch.zeros(B, A, device=root_states.device))
        e["s"].copy_(root_states); e["v"].copy_(root_values.float()); e["l"].copy_(root_logits.float()); e["g"].copy_(g)
        e["graph"].replay()
        value, policy, best, overflow = e["out"]
        if bool(overflow):  # deeper than the bound: exact eager search with the same noise
            self.fallbacks += 1
            return self.mcts.search(self.model, root_states, root_values, root_logits, g=g)
        return value.cpu().numpy(), policy.cpu().numpy(), best.cpu().numpy()
