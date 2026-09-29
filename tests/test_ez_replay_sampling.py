"""EZReplay's prioritised sampling matches EZ-V2's replay buffer; uniform sampling is untouched.

EZ-V2 (ez/data/replay_buffer.py) samples a batch without replacement, clips normalised IS weights to
[0.1, 1], and gives new data the buffer's *current* max priority. Our port drew with replacement, left the
weights unclipped and used a running maximum that never came down - the diff S-067 asked for.
"""
import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture(scope="module")
def ez():
    """Imports the trainer, stubbing the two gym-dependent modules when gym is not installed (this PC)."""
    stubs = {}
    try:
        import gym  # noqa: F401
    except Exception:
        for name, attrs in (("atari_qwen.envs.atari_wrappers", ("make_vector_atari_envs", "make_atari_env")),
                            ("atari_qwen.eval.evaluate", ("compute_hns",))):
            if name not in sys.modules:
                mod = types.ModuleType(name)
                for a in attrs:
                    setattr(mod, a, lambda *args, **kw: None)
                stubs[name] = sys.modules[name] = mod
    try:
        yield importlib.import_module("atari_qwen.training.train_ez_offpolicy")
    finally:
        for name in stubs:
            sys.modules.pop(name, None)


def _add_row(r):
    import torch
    E = r.E
    # Frames as a torch tensor: add() goes through torch.as_tensor, which cannot read numpy where torch and
    # numpy disagree on ABI (this PC); the replay logic under test is the same either way.
    r.add(torch.zeros((E, 3, 8, 8), dtype=torch.uint8), np.zeros(E, np.int64), np.zeros(E, np.float32),
          np.zeros(E, bool), np.full((E, 4), 0.25, np.float32), np.zeros(E, np.float32))


def _filled(ez, alpha, T=200, E=4, prio=None, rows=None):
    r = ez.EZReplay(T, E, (3, 8, 8), action_dim=4, alpha=alpha)
    for _ in range(T if rows is None else rows):
        _add_row(r)
    if prio is not None:
        r.prio[:r.size] = prio
    return r


def test_prioritised_batch_has_no_repeats_and_clipped_weights(ez):
    rng = np.random.default_rng(0)
    prio = np.full((200, 4), 1e-3)
    prio[rng.integers(0, 190, 10), rng.integers(0, 4, 10)] = 5.0   # a few reward windows dominate the mass
    r = _filled(ez, alpha=1.0, prio=prio)
    np.random.seed(0)
    t, e, w = r.sample(256, horizon=10)
    flat = t * r.E + e
    assert len(np.unique(flat)) == 256, "EZ-V2 samples a batch without replacement"
    assert w.min() >= 0.1 - 1e-7 and w.max() <= 1.0 + 1e-7


def test_new_data_enters_at_the_current_not_the_historical_max(ez):
    r = _filled(ez, alpha=1.0, T=60, rows=50, prio=np.full((50, 4), 0.5))
    r.update_priorities(np.array([3]), np.array([0]), np.array([100.0]))   # one outlier error ...
    r.update_priorities(np.array([3]), np.array([0]), np.array([0.01]))    # ... corrected on its next visit
    _add_row(r)
    assert np.allclose(r.prio[r.size - 1], 0.5), "the corrected outlier must not set the entry priority"


def test_uniform_sampling_draw_is_unchanged(ez):
    """alpha = 0 keeps the original with-replacement draw, so S058f/g seeds reproduce exactly."""
    r = _filled(ez, alpha=0.0)
    np.random.seed(123)
    t, e, w = r.sample(256, horizon=10)
    n = r.size - 10
    p = np.full(n * r.E, 1.0 / (n * r.E))
    np.random.seed(123)
    ref = np.random.choice(p.size, size=256, p=p)
    assert np.array_equal(t * r.E + e, ref)
    assert np.allclose(w, 1.0)
