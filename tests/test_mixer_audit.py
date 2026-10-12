"""TODO B37 audit: zero-initialised mixer tensors must not report ~1e12 'relative change' (S-127)."""
import math

import torch
import torch.nn as nn

from atari_qwen.models.ez_model import TokenMixer
from atari_qwen.training import mixer_audit


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.repr_mixer = TokenMixer(channels=8, hw=4, dim=16, depth=1, heads=2)
        self.head = nn.Linear(8, 2)


def test_every_mixer_tensor_has_a_category():
    net = _Net()
    cats = {mixer_audit.category(n) for n, _ in net.named_parameters() if "mixer" in n}
    assert "other" not in cats
    assert {"io", "norm"} <= cats


def test_snapshot_holds_only_mixer_tensors():
    w0 = mixer_audit.snapshot(_Net())
    assert w0 and all("mixer" in n for n in w0)


def test_zero_init_tensor_stays_bounded_and_absolute_change_is_exact():
    torch.manual_seed(0)
    net = _Net()
    w0 = mixer_audit.snapshot(net)
    assert float(net.repr_mixer.out.weight.detach().norm()) == 0.0         # the zero-init output projection: the case that gave |dW| / 1e-12
    n_io = sum(1 for n in w0 if mixer_audit.category(n) == "io")
    with torch.no_grad():
        net.repr_mixer.out.weight.add_(0.5)
    rel, grad, ab = mixer_audit.audit(net, w0)
    moved = 0.5 * math.sqrt(8 * 16)                                # |dW| of out.weight
    assert abs(ab["io"] - moved / n_io) < 1e-4                     # mean over the io tensors; only out.weight moved
    assert abs(rel["io"] - moved / mixer_audit.REL_FLOOR / n_io) < 1e-1
    assert rel["io"] < 1e5                                          # the old definition gave ~1e12 here
    assert ab["ffn"] == 0.0 and rel["qkv"] == 0.0
    assert grad == {}                                              # no backward pass yet


def test_gradient_norms_are_reported_per_category():
    torch.manual_seed(0)
    net = _Net()
    w0 = mixer_audit.snapshot(net)
    net.repr_mixer.out.weight.data.normal_(0, 0.1)                 # open the output so gradient reaches the blocks
    net.repr_mixer(torch.randn(3, 8, 2, 2)).pow(2).sum().backward()
    rel, grad, ab = mixer_audit.audit(net, w0)
    assert grad["io"] > 0 and grad["ffn"] > 0
    assert all(v >= 0 for d in (rel, grad, ab) for v in d.values())
