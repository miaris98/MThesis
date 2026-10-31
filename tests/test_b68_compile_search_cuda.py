"""TODO B68 E1: the compiled target recurrent_inference used inside the graphed reanalysis search (CUDA only).

It must compute what the eager function computes (up to bf16 rounding) and follow a target refresh - load_state_dict copies the
weights in place - without a recompile: a stale compiled copy would silently search with an old network. Whole-search agreement is
measured on real checkpoints instead (atari_qwen/scripts/b68_compile_check.py: compiled and eager are equally close to an fp32
search), because on a random network one rounding flip in the tree compounds."""
import pytest
import torch

from atari_qwen.models.ez_model import DiscreteSupport, EZV2Model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _model(seed):
    torch.manual_seed(seed)
    m = EZV2Model(4, obs_channels=12, support=DiscreteSupport(), trunk="qwen", mixer_size="small", mixer_qk_norm=True, state_hw=6)
    with torch.no_grad():   # a fresh model's heads and mixer output are zero-initialised: small perturbations make outputs depend on the weights
        for p in m.parameters():
            p.add_(0.02 * torch.randn_like(p))
    return m.cuda().eval()


def _flat(out):
    return [t.float() for x in out for t in (x if isinstance(x, tuple) else (x,))]


def _rel(xs, ys):
    return max(float((x - y).norm() / y.norm().clamp_min(1e-6)) for x, y in zip(xs, ys))


def test_compiled_recurrent_inference_matches_eager_and_follows_a_target_refresh():
    eager, comp = _model(0), _model(0)
    comp.recurrent_inference = torch.compile(comp.recurrent_inference, dynamic=False)
    g = torch.Generator("cuda").manual_seed(0)
    s = torch.randn(96, 64, 6, 6, device="cuda", generator=g)
    a = torch.randint(0, 4, (96,), device="cuda", generator=g)
    h = (torch.zeros(1, 96, eager.lstm_hidden, device="cuda"), torch.zeros(1, 96, eager.lstm_hidden, device="cuda"))

    def run(m):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return _flat(m.recurrent_inference(s, a, h))

    e1, c1 = run(eager), run(comp)
    assert _rel(c1, e1) < 0.03                        # same function, bf16 rounding only
    new = _model(1).state_dict()                      # a target refresh: new weights copied in place
    eager.load_state_dict(new); comp.load_state_dict(new)
    e2, c2 = run(eager), run(comp)
    assert _rel(e2, e1) > 0.2                         # the refresh changed the outputs ...
    assert _rel(c2, e2) < 0.03                        # ... and the compiled copy followed it
