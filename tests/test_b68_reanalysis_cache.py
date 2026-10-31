"""TODO B68: the per-target-window reanalysis cache must give exactly the targets a fresh search with the same (hashed) noise gives.

Mock functions check the bookkeeping exactly (hits, misses, padding, versions, unneeded roots, bootstrap sharing); the last test runs the
real Gumbel search on a small Qwen-mixer model on CPU and compares cached against uncached targets over overlapping batches."""
import math

import numpy as np
import pytest
import torch

from atari_qwen.mcts.gumbel_mcts import GumbelMCTS
from atari_qwen.models.ez_model import DiscreteSupport, EZV2Model
from atari_qwen.training.reanalysis_cache import BUCKETS, ReanalysisCache, bucket, hashed_gumbel

A = 4


def test_hashed_gumbel_is_a_deterministic_standard_gumbel():
    keys = np.arange(50_000)
    g = hashed_gumbel(keys, 3, A, "cpu")
    assert g.shape == (50_000, A) and torch.isfinite(g).all()
    assert torch.equal(g, hashed_gumbel(keys, 3, A, "cpu"))
    assert not torch.equal(g, hashed_gumbel(keys, 4, A, "cpu"))
    assert abs(float(g.mean()) - 0.5772) < 0.01 and abs(float(g.std()) - math.pi / math.sqrt(6)) < 0.01   # Euler gamma, pi/sqrt(6)
    assert float((g[:, 0] - g[:, 1]).abs().mean()) > 0.5          # actions are not copies of each other
    assert torch.equal(hashed_gumbel(keys[17:19], 3, A, "cpu"), g[17:19])   # a key's noise does not depend on the batch
    if torch.cuda.is_available():
        assert torch.equal(hashed_gumbel(keys, 3, A, "cuda").cpu(), g)


def test_bucket_sizes():
    assert [bucket(n) for n in (1, 96, 97, 1536)] == [96, 96, 192, 1536] and BUCKETS[-1] == 1536


class Mock:
    """stacks / initial inference / search as pure functions of the key and the noise, recording what was computed."""

    def __init__(self):
        self.searched, self.inferred, self.batch_sizes = [], [], []

    def stacks(self, keys):
        return torch.as_tensor(np.asarray(keys), dtype=torch.float32)[:, None]

    def ii(self, x):
        self.inferred.extend(x[:, 0].long().tolist())
        return x, x[:, 0] * 0.1, torch.zeros(len(x), A)

    def search(self, s, v, logits, g):
        self.batch_sizes.append(len(s)); self.searched.extend(s[:, 0].long().tolist())
        return (v + 0.01 * g.sum(1)).numpy(), torch.softmax(g, 1).numpy()


def reference(keys, need, ver):
    g = hashed_gumbel(keys, ver, A, "cpu")
    v = torch.as_tensor(keys, dtype=torch.float32) * 0.1
    return np.where(need, (v + 0.01 * g.sum(1)).numpy(), 0), torch.softmax(g, 1).numpy() * need[:, None]


def test_cache_returns_the_fresh_search_and_reuses_it():
    rng = np.random.default_rng(0)
    c, m = ReanalysisCache(5_000, A, "cpu"), Mock()
    for step in range(6):
        keys = rng.integers(0, 400, size=300)                       # heavy overlap between steps, like a late buffer
        need = rng.random(300) > 0.1
        bkeys = rng.integers(0, 400, size=300); bneed = rng.random(300) > 0.5
        sv, spi, vb = c.reanalyse(keys, need, bkeys, bneed, 2, m.stacks, m.ii, m.search, A)
        rv, rpi = reference(keys, need, 2)
        np.testing.assert_allclose(sv, rv, rtol=0, atol=1e-6); np.testing.assert_allclose(spi, rpi, rtol=0, atol=1e-6)
        np.testing.assert_allclose(vb, np.where(bneed, (torch.as_tensor(bkeys, dtype=torch.float32) * 0.1).numpy(), 0), rtol=0, atol=1e-6)
    # within one version a position is inferred at most twice: once for its value (bootstrap) and, if it is drawn as a root later,
    # once more for the search, which needs the latent state (not cached: ~460 MB at 100k positions for one cheap pass)
    assert max(np.unique(m.inferred, return_counts=True)[1]) <= 2
    assert set(m.batch_sizes) <= set(BUCKETS)                       # every search ran at a bucket size (one CUDA graph each)
    hit, vhit = c.stats()
    assert 0.5 < hit < 1.0 and 0.0 < vhit < 1.0


def test_unneeded_roots_are_not_searched_and_a_new_version_recomputes():
    c, m = ReanalysisCache(1_000, A, "cpu"), Mock()
    keys = np.array([5, 6, 7, 8]); need = np.array([True, False, True, False])
    c.reanalyse(keys, need, keys[:0], need[:0], 0, m.stacks, m.ii, m.search, A)
    assert set(m.searched) == {5, 7}
    sv0, _, _ = c.reanalyse(keys, need, keys[:0], need[:0], 0, m.stacks, m.ii, m.search, A)
    n_before = len(m.searched)
    sv1, _, _ = c.reanalyse(keys, need, keys[:0], need[:0], 1, m.stacks, m.ii, m.search, A)   # target refreshed
    assert len(m.searched) > n_before and not np.allclose(sv0, sv1)                         # new noise, new search
    assert sv1[1] == 0 and sv1[3] == 0


def test_bootstrap_values_reuse_the_root_pass():
    c, m = ReanalysisCache(1_000, A, "cpu"), Mock()
    roots = np.array([10, 11, 12]); boots = np.array([11, 12, 13])
    c.reanalyse(roots, np.ones(3, bool), boots, np.ones(3, bool), 0, m.stacks, m.ii, m.search, A)
    # 10-12 went through initial inference once for the search; 11 and 12 reuse it as bootstrap values, only 13 needs its own pass
    assert sorted(m.inferred) == [10, 11, 12, 13]
    _, vhit = c.stats()
    assert abs(vhit - 2 / 3) < 1e-9


def _batches(rng, n=4, size=300, span=400):
    return [(rng.integers(0, span, size), rng.random(size) > 0.1, rng.integers(0, span, size), rng.random(size) > 0.5)
            for _ in range(n)]


def test_merged_fill_gives_each_update_its_own_targets_in_fewer_searches():
    """TODO B68 M1: the misses of the updates between two env steps searched in one fill call; each update then reads exactly the
    targets it would have computed itself and the same positions are searched. Bootstrap passes can only drop: per update, a position
    that is a bootstrap state of one update and a root of a later one gets a value-only pass first; merged, the root search covers it."""
    batches = _batches(np.random.default_rng(3))
    per, m1 = ReanalysisCache(5_000, A, "cpu"), Mock()
    want = [per.reanalyse(*b, 2, m1.stacks, m1.ii, m1.search, A) for b in batches]
    merged, m2 = ReanalysisCache(5_000, A, "cpu"), Mock()
    merged.fill(*(np.concatenate([b[i] for b in batches]) for i in range(4)), 2, m2.stacks, m2.ii, m2.search, A)
    calls, inferred = len(m2.batch_sizes), len(m2.inferred)
    got = [merged.reanalyse(*b, 2, m2.stacks, m2.ii, m2.search, A) for b in batches]
    assert len(m2.batch_sizes) == calls and len(m2.inferred) == inferred   # the per-update calls only read the cache
    for w, g in zip(want, got):
        for x, y in zip(w, g):
            np.testing.assert_array_equal(x, y)
    assert set(m2.searched) == set(m1.searched) and calls < len(m1.batch_sizes)
    assert set(m2.batch_sizes) <= set(BUCKETS)
    (hit_p, vhit_p), (hit_m, vhit_m) = per.stats(), merged.stats()
    assert hit_m == hit_p and vhit_m >= vhit_p and len(m2.inferred) < len(m1.inferred)


def test_merged_fill_matches_per_update_on_the_real_search():
    torch.manual_seed(0)
    sup = DiscreteSupport()
    model = EZV2Model(A, obs_channels=12, support=sup, trunk="qwen", mixer_size="small", mixer_qk_norm=True, state_hw=6).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.repr_mixer.out.weight, std=0.02); torch.nn.init.normal_(model.dyn_mixer.out.weight, std=0.02)
    mcts = GumbelMCTS(A, sup, num_simulations=8, discount=0.997 ** 4, lstm_horizon=5)
    obs = torch.rand(80, 12, 96, 96, generator=torch.Generator().manual_seed(4))

    def ii(x):
        s, v, p = model.initial_inference(x)
        return s, sup.vector_to_scalar(v), p

    def search(s, v, p, g):
        out = mcts.search(model, s, v, p, g=g)
        return out[0], out[1]

    stacks = lambda keys: obs[torch.as_tensor(np.asarray(keys))]
    batches = _batches(np.random.default_rng(5), n=3, size=24, span=80)
    with torch.no_grad():
        per = ReanalysisCache(80, A, "cpu")
        want = [per.reanalyse(*b, 1, stacks, ii, search, A) for b in batches]
        merged = ReanalysisCache(80, A, "cpu")
        merged.fill(*(np.concatenate([b[i] for b in batches]) for i in range(4)), 1, stacks, ii, search, A)
        got = [merged.reanalyse(*b, 1, stacks, ii, search, A) for b in batches]
    for w, g in zip(want, got):
        for x, y in zip(w, g):
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-5)    # fp32; only the batch composition of the calls differs


def test_real_search_cached_equals_uncached_with_the_same_noise():
    torch.manual_seed(0)
    sup = DiscreteSupport()
    model = EZV2Model(A, obs_channels=12, support=sup, trunk="qwen", mixer_size="small", mixer_qk_norm=True, state_hw=6).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.repr_mixer.out.weight, std=0.02); torch.nn.init.normal_(model.dyn_mixer.out.weight, std=0.02)
    mcts = GumbelMCTS(A, sup, num_simulations=8, discount=0.997 ** 4, lstm_horizon=5)
    obs = torch.rand(60, 12, 96, 96, generator=torch.Generator().manual_seed(1))   # 60 replay positions

    def ii(x):
        s, v, p = model.initial_inference(x)
        return s, sup.vector_to_scalar(v), p

    def search(s, v, p, g):
        out = mcts.search(model, s, v, p, g=g)
        return out[0], out[1]

    stacks = lambda keys: obs[torch.as_tensor(np.asarray(keys))]
    c = ReanalysisCache(60, A, "cpu")
    rng = np.random.default_rng(2)
    with torch.no_grad():
        for _ in range(3):
            keys = rng.integers(0, 60, size=24); need = np.ones(24, bool)
            sv, spi, _ = c.reanalyse(keys, need, keys[:0], need[:0], 7, stacks, ii, search, A)
            s, v, p = ii(stacks(keys))
            rv, rpi, _ = mcts.search(model, s, v, p, g=hashed_gumbel(keys, 7, A, "cpu"))
            np.testing.assert_allclose(sv, rv, rtol=0, atol=1e-5)
            np.testing.assert_allclose(spi, rpi, rtol=0, atol=1e-5)
    assert c.stats()[0] > 0
