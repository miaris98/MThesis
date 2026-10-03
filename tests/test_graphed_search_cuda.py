"""GraphedSearch inside training (TODO B23 step 1b): the captured search must follow in-place weight updates (the optimizer
steps `model`, `target.load_state_dict` overwrites the target) and match the eager search exactly, also under bf16 autocast
(the reanalyze path). CUDA only; skipped elsewhere."""
import copy

import numpy as np
import pytest
import torch

from atari_qwen.mcts.gumbel_mcts import GraphedSearch, GumbelMCTS
from atari_qwen.models.ez_model import DiscreteSupport, EZV2Model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


def _model(trunk="resnet", seed=0):
    torch.manual_seed(seed)
    m = EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk=trunk, state_hw=6).cuda().eval()
    _perturb(m, 0.05)
    return m


def _perturb(m, scale):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(scale * torch.randn_like(p))
        for b in m.buffers():  # BatchNorm running stats move in training too
            if b.dtype.is_floating_point and b.dim() > 0:
                b.mul_(1.0 + 0.01 * torch.randn_like(b))


def _mcts(support, sims=16):
    return GumbelMCTS(4, support, num_simulations=sims, discount=0.997 ** 4, lstm_horizon=5)


def _roots(model, x, amp):
    with torch.no_grad(), torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
        s, v, p = model.initial_inference(x)
    return s, model.support.vector_to_scalar(v), p


def _same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("trunk", ["resnet", "gtrxl"])
def test_acting_graph_follows_optimizer_steps(trunk):
    model = _model(trunk)
    mcts, gs = _mcts(model.support), GraphedSearch(_mcts(model.support), model)
    x = torch.rand(4, 12, 96, 96, device="cuda")
    for i in range(4):
        s, v, p = _roots(model, x, None)
        torch.manual_seed(i); eager = mcts.search(model, s, v, p, add_noise=True)
        torch.manual_seed(i); graphed = gs(s, v, p, True)
        assert _same(eager, graphed), f"step {i}"
        _perturb(model, 0.01)  # an optimizer step: parameters change in place, the graph must see it
    assert gs.fallbacks == 0


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False),
                    reason="needs native bf16")
@pytest.mark.parametrize("trunk", ["resnet", "gtrxl"])
def test_reanalyze_graph_under_bf16_follows_target_updates(trunk):
    model = _model(trunk)
    target = copy.deepcopy(model).eval()
    mcts = _mcts(model.support)
    gs = GraphedSearch(_mcts(model.support), target, amp_dtype=torch.bfloat16)
    x = torch.rand(36, 12, 96, 96, device="cuda")
    for i in range(3):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # as the trainer's targets block
            s, v, p = _roots(target, x, torch.bfloat16)
            torch.manual_seed(i); eager = mcts.search(target, s, v, p, add_noise=True)
            torch.manual_seed(i); graphed = gs(s, v, p, True)
        assert _same(eager, graphed), f"target version {i}"
        _perturb(model, 0.02)
        target.load_state_dict(model.state_dict())  # the hard target update every 200 updates
    assert gs.fallbacks == 0
