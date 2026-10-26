"""TODO B68 (2026-10-26): per-target-window cache of the reanalysis targets.

Every target of an update comes from the target network, which changes only every `--target-update-interval` updates (200, as EZ-V2's
code). Inside one such window a replay state is searched again each time it is drawn - 3 to 30 times over a 100k run - and only the
Gumbel noise differs between those searches (docs/ops/2026-10-26_reanalysis_cost_analysis.md: one 16-sim search differs from another
by policy TV 0.063 mean / 0.005 median and value 0.004). This cache searches each replay position once per window and serves the
repeats from memory: 19% of today's searches over a 100k run at W = 200.

Exact by construction: the noise of a position is a hash of (position, window), so a cached result equals what a fresh search would
compute in that window, before or after a resume (the cache itself is not checkpointed; the replay never overwrites a position in our
runs, capacity >= run length). The only change against the uncached trainer is "one noise draw per (state, window)" instead of "one per
draw". Root and bootstrap passes are the same function (target.initial_inference on a position's stack), so they share one value cache.
"""
from typing import Callable, Tuple

import numpy as np
import torch

BUCKETS = (96, 192, 384, 768, 1536)  # miss batches are padded to these sizes: one CUDA graph each instead of one per miss count
_M1, _M2, _M3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def _as_signed(c: int) -> int:
    return c - (1 << 64) if c >= 1 << 63 else c


def _lshr(z: torch.Tensor, sh: int) -> torch.Tensor:
    """Logical right shift of int64 bit patterns: torch's >> is arithmetic (copies the sign bit), splitmix64 needs zeros shifted in."""
    return (z >> sh) & ((1 << (64 - sh)) - 1)


def hashed_gumbel(keys: np.ndarray, version: int, num_actions: int, device) -> torch.Tensor:
    """Standard Gumbel noise (n, A), a pure function of (replay position, target version, action): splitmix64, whose uint64
    arithmetic is reproduced exactly by wrapping int64 products, xor and logical shifts."""
    k = torch.as_tensor(np.asarray(keys, dtype=np.int64), device=device)[:, None]
    a = torch.arange(num_actions, device=device, dtype=torch.int64)[None, :]
    ver = torch.tensor(int(version), device=device, dtype=torch.int64)   # a tensor, so the product wraps in int64 like the rest
    z = k * _as_signed(_M1) + ver * _as_signed(_M2) + (a + 1) * _as_signed(_M3)
    for sh, mul in ((30, _M2), (27, _M3)):
        z = (z ^ _lshr(z, sh)) * _as_signed(mul)
    z = z ^ _lshr(z, 31)
    u = _lshr(z, 11).to(torch.float64).add_(0.5).div_(float(1 << 53))   # the top 53 bits, in (0, 1), never 0 or 1
    return (-torch.log(-torch.log(u))).to(torch.float32)


def bucket(n: int) -> int:
    for b in BUCKETS:
        if n <= b:
            return b
    return n


class ReanalysisCache:
    """Search value / improved policy and initial-inference value per flat replay position (row * E + env), tagged with the target
    version they were computed under. Lives on the replay's device; ~3 MB for a 100k-transition buffer."""

    def __init__(self, num_positions: int, num_actions: int, device):
        self.device = device
        self.s_ver = torch.full((num_positions,), -1, dtype=torch.int64, device=device)
        self.s_val = torch.zeros(num_positions, dtype=torch.float32, device=device)
        self.s_pi = torch.zeros(num_positions, num_actions, dtype=torch.float32, device=device)
        self.v_ver = torch.full((num_positions,), -1, dtype=torch.int64, device=device)
        self.v_val = torch.zeros(num_positions, dtype=torch.float32, device=device)
        self.hits = self.lookups = self.v_hits = self.v_lookups = 0

    def stats(self, reset: bool = True) -> Tuple[float, float]:
        r = (self.hits / max(1, self.lookups), self.v_hits / max(1, self.v_lookups))
        if reset:
            self.hits = self.lookups = self.v_hits = self.v_lookups = 0
        return r

    def reanalyse(self, root_keys: np.ndarray, root_need: np.ndarray, boot_keys: np.ndarray, boot_need: np.ndarray, version: int,
                  stacks: Callable, initial_inference: Callable, search: Callable, num_actions: int):
        """Targets for one update.

        root_keys / root_need: (n_root,) flat positions of the search roots and whether any loss uses them (False past an episode end:
        those targets are zeroed by the caller anyway, so they are not searched). boot_keys / boot_need: the same for the bootstrap values.
        stacks(keys) -> observation stacks; initial_inference(x) -> (latent, scalar value, policy logits) of the target network;
        search(latent, value, logits, g) -> (values (n,), policies (n, A)) as numpy, with the Gumbel noise g given.
        Returns (search values (n_root,), policies (n_root, A), bootstrap values (n_boot,)) as numpy; unneeded entries are 0."""
        dev = self.device
        # 1. roots: one search per distinct needed position not yet searched under this version
        uk = np.unique(root_keys[root_need])
        if len(uk):
            ukt = torch.as_tensor(uk, device=dev)
            miss = uk[(self.s_ver[ukt] != version).cpu().numpy()]
            self.lookups += len(uk); self.hits += len(uk) - len(miss)
            for lo in range(0, len(miss), BUCKETS[-1]):
                mk = miss[lo:lo + BUCKETS[-1]]
                n = len(mk); pad = bucket(n) - n
                s, v, logits = initial_inference(stacks(mk))
                # only the search needs a fixed shape (its CUDA graph): pad its inputs with copies of row 0, drop their outputs
                rep = lambda x: torch.cat([x, x[:1].expand(pad, *x.shape[1:])]) if pad else x
                g = hashed_gumbel(np.concatenate([mk, np.repeat(mk[:1], pad)]), version, num_actions, dev)
                sv, spi = search(rep(s), rep(v), rep(logits), g)
                kt = torch.as_tensor(mk, device=dev)
                self.s_val[kt] = torch.as_tensor(np.asarray(sv[:n], np.float32), device=dev)
                self.s_pi[kt] = torch.as_tensor(np.asarray(spi[:n], np.float32), device=dev)
                self.s_ver[kt] = version
                self.v_val[kt] = v[:n].float()
                self.v_ver[kt] = version
        rk = torch.as_tensor(root_keys, device=dev)
        need = torch.as_tensor(root_need, device=dev)
        sv_all = torch.where(need, self.s_val[rk], torch.zeros((), device=dev))
        pi_all = self.s_pi[rk] * need[:, None]
        # 2. bootstrap values: initial inference only, shared with the root pass of the same position and version
        ub = np.unique(boot_keys[boot_need])
        if len(ub):
            ubt = torch.as_tensor(ub, device=dev)
            vmiss = ub[(self.v_ver[ubt] != version).cpu().numpy()]
            self.v_lookups += len(ub); self.v_hits += len(ub) - len(vmiss)
            for lo in range(0, len(vmiss), BUCKETS[-1]):
                mk = vmiss[lo:lo + BUCKETS[-1]]
                kt = torch.as_tensor(mk, device=dev)
                _, v, _ = initial_inference(stacks(mk))
                self.v_val[kt] = v.float()
                self.v_ver[kt] = version
        bk = torch.as_tensor(boot_keys, device=dev)
        vb_all = torch.where(torch.as_tensor(boot_need, device=dev), self.v_val[bk], torch.zeros((), device=dev))
        return sv_all.cpu().numpy(), pi_all.cpu().numpy(), vb_all.cpu().numpy()
