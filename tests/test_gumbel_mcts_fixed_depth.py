"""GumbelMCTS.search(depth_bound=...) (the sync-free mode GraphedSearch captures) must equal the default search exactly
unless a tree is deeper than the bound, which it must report (TODO B23)."""
import torch

from atari_qwen.mcts.gumbel_mcts import GumbelMCTS
from atari_qwen.models.ez_model import DiscreteSupport, EZV2Model


def _setup(sims, B=6, seed=0):
    torch.manual_seed(seed)
    support = DiscreteSupport()
    model = EZV2Model(4, obs_channels=12, support=support, trunk="resnet", state_hw=6).eval()
    with torch.no_grad():  # the heads start at zero: perturb everything so the search trees are not trivial
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
        x = torch.rand(B, 12, 96, 96)
        s, v, p_logits = model.initial_inference(x)
    mcts = GumbelMCTS(4, support, num_simulations=sims, discount=0.997 ** 4, lstm_horizon=5)
    return model, mcts, s, support.vector_to_scalar(v), p_logits


def test_fixed_depth_matches_default_search():
    for sims in (8, 16):
        model, mcts, s, v, p = _setup(sims)
        g = torch.distributions.Gumbel(0.0, 1.0).sample((s.shape[0], 4))
        rv, rp, ra = mcts.search(model, s, v, p, g=g)
        fv, fp, fa, overflow = mcts.search(model, s, v, p, g=g, depth_bound=sims)  # a tree can never be deeper than sims
        assert not bool(overflow)
        assert (fv.numpy() == rv).all() and (fp.numpy() == rp).all() and (fa.numpy() == ra).all()


def test_too_small_a_bound_is_reported():
    model, mcts, s, v, p = _setup(16)
    g = torch.zeros(s.shape[0], 4)
    *_, overflow = mcts.search(model, s, v, p, g=g, depth_bound=1)
    default = mcts.search(model, s, v, p, g=g)
    deeper = mcts.search(model, s, v, p, g=g, depth_bound=16)
    assert not bool(deeper[3])
    assert bool(overflow), "a depth bound of 1 cannot hold a 16-simulation tree"
    assert default[2].shape == (s.shape[0],)
