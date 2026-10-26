"""TODO B67: the QK-norm option of QwenTransformerBlock (Qwen3-style per-head RMSNorm on queries and keys), on both benchmarks.

Off (the default) must be the original block exactly - same tensors, same outputs - so E15 and every existing checkpoint are untouched.
On, the attention logits must not grow with the q/k weights (the B66 collapse: logits ~9,000 and one-hot attention by update 8,004
under AdamW 3e-4), the CARLA policy must round-trip through the checkpoint config, a mismatch must fail loudly, and both recipes must
leave the new gains out of weight decay (CARLA's wor_optim and the Atari mixer_param_groups both exclude ndim <= 1)."""
import math

import pytest
import torch
import torch.nn.functional as F

from atari_qwen.models.ez_model import EZV2Model
from atari_qwen.training.train_ez_offpolicy import mixer_param_groups
from src.models.transformer.layers import QwenTransformerBlock
from src.models.world_on_rails import QwenWorldOnRailsPolicy
from src.models.world_on_rails.wor_loader import load_wor_model
from src.training.wor_trainer import WorldOnRailsTrainer

POLICY = dict(backbone_name="resnet34", pretrained=False, route_points=4, model_size="10m", vision_grid=4)


def _reference_attention(att, x):
    """The pre-B67 forward of QwenAttentionWithSkip, written out by hand."""
    B, N, C = x.shape
    q = att.q_proj(x).view(B, N, att.num_heads, att.head_dim).transpose(1, 2)
    k = att.k_proj(x).view(B, N, att.num_heads, att.head_dim).transpose(1, 2)
    v = att.v_proj(x).view(B, N, att.num_heads, att.head_dim).transpose(1, 2)
    return att.out_proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous().view(B, N, C))


def _logits(att, x):
    B, N, C = x.shape
    q = att.q_proj(x).view(B, N, att.num_heads, att.head_dim).transpose(1, 2)
    k = att.k_proj(x).view(B, N, att.num_heads, att.head_dim).transpose(1, 2)
    if att.q_norm is not None:
        q, k = att.q_norm(q), att.k_norm(k)
    return (q @ k.transpose(-2, -1)) * att.scale


def test_default_block_is_the_original():
    torch.manual_seed(0)
    blk = QwenTransformerBlock(64, num_heads=4, ffn_dim=128)
    assert blk.attn.q_norm is None and blk.attn.k_norm is None
    assert not any("q_norm" in k or "k_norm" in k for k in blk.state_dict())
    x = torch.randn(2, 36, 64)
    assert torch.equal(blk.attn(x), _reference_attention(blk.attn, x))


def test_qk_norm_adds_one_gain_per_head_dim_and_bounds_the_logits():
    torch.manual_seed(0)
    blk = QwenTransformerBlock(64, num_heads=4, ffn_dim=128, qk_norm=True)
    extra = {k: tuple(v.shape) for k, v in blk.state_dict().items() if k.startswith("attn.") and "_norm" in k}
    assert extra == {"attn.q_norm.weight": (16,), "attn.k_norm.weight": (16,)}
    x = torch.randn(2, 36, 64)
    with torch.no_grad():
        blk.attn.q_proj.weight.mul_(1000.0)
        blk.attn.k_proj.weight.mul_(1000.0)   # the B66 weight growth, exaggerated
        # |q_hat| = |k_hat| <= sqrt(d) at unit gains, so |logit| <= sqrt(d) * sqrt(d) / sqrt(d)
        assert float(_logits(blk.attn, x).abs().max()) <= math.sqrt(blk.attn.head_dim) * (1 + 1e-4)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_only_qk_norm_makes_the_block_ignore_the_scale_of_the_qk_weights(qk_norm):
    torch.manual_seed(0)
    blk = QwenTransformerBlock(64, num_heads=4, ffn_dim=128, qk_norm=qk_norm).eval()
    x = torch.randn(2, 36, 64)
    with torch.no_grad():
        y0 = blk(x)
        blk.attn.q_proj.weight.mul_(30.0)
        blk.attn.k_proj.weight.mul_(30.0)
        y1 = blk(x)
    assert torch.allclose(y0, y1, atol=1e-4) == qk_norm


def test_bf16_autocast_keeps_one_dtype_for_sdpa():
    blk = QwenTransformerBlock(64, num_heads=4, ffn_dim=128, qk_norm=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = blk(torch.randn(2, 36, 64))
    assert torch.isfinite(y.float()).all()


def test_default_carla_policy_has_no_new_tensors():
    m = QwenWorldOnRailsPolicy(**POLICY)
    assert m.qk_norm is False
    assert not any("q_norm" in k or "k_norm" in k for k in m.state_dict())


def test_carla_trainer_stamps_qk_norm_and_keeps_its_gains_out_of_weight_decay(tmp_path):
    policy = QwenWorldOnRailsPolicy(**POLICY, qk_norm=True)
    trainer = WorldOnRailsTrainer(model=policy, data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"), batch_size=4,
                                  num_workers=0, device="cpu", synthetic_samples=8, use_mlflow=False)
    assert trainer.model_config["qk_norm"] is True
    gains = {id(p) for n, p in policy.named_parameters() if ".q_norm." in n or ".k_norm." in n}
    assert len(gains) == 2 * len(policy.trunk.blocks)
    for group in trainer.optimizer.param_groups:
        if any(id(p) in gains for p in group["params"]):
            assert group["weight_decay"] == 0.0


def _save(tmp_path, model, stamp_qk):
    path = tmp_path / "ckpt.pth"
    cfg = {"backbone": "resnet34", "route_points": 4, "vision_grid": [4, 4], "qk_norm": stamp_qk}
    torch.save({"epoch": 1, "model": model.state_dict(), "config": cfg}, path)
    return str(path)


def test_carla_checkpoint_rebuilds_with_qk_norm(tmp_path, capsys):
    m = QwenWorldOnRailsPolicy(**POLICY, qk_norm=True).eval()
    path = _save(tmp_path, m, True)
    capsys.readouterr()
    loaded = load_wor_model(checkpoint_path=path, backbone_name="resnet34", pretrained_backbone=False, device="cpu",
                            policy_arch="qwen10m", route_points=4).eval()
    out = capsys.readouterr().out
    assert "Missing keys: 0" in out and "Unexpected keys: 0" in out, out
    assert loaded.qk_norm and all(b.attn.q_norm is not None for b in loaded.trunk.blocks)
    rgb = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(3))
    speed, cmd, route = torch.tensor([[5.0]]), torch.tensor([3]), torch.zeros(1, 4, 2)
    with torch.no_grad():
        pa, pb = m(rgb, speed, cmd, route), loaded(rgb, speed, cmd, route)
    for k in pa:
        assert torch.allclose(pa[k], pb[k], atol=1e-6), k


def test_carla_qk_norm_mismatch_is_fatal(tmp_path):
    m = QwenWorldOnRailsPolicy(**POLICY, qk_norm=True)
    path = _save(tmp_path, m, False)   # weights with norms, config without: strict=False would silently drop them
    with pytest.raises(RuntimeError, match="QK-norm mismatch"):
        load_wor_model(checkpoint_path=path, backbone_name="resnet34", pretrained_backbone=False, device="cpu",
                       policy_arch="qwen10m", route_points=4)


def test_atari_mixer_qk_norm_blocks_and_param_groups():
    m = EZV2Model(4, obs_channels=12, trunk="qwen", mixer_size="small", mixer_qk_norm=True).eval()
    blocks = list(m.repr_mixer.blocks) + list(m.dyn_mixer.blocks)
    assert blocks and all(b.attn.q_norm is not None and b.attn.k_norm is not None for b in blocks)
    names = {id(p): n for n, p in m.named_parameters()}
    decay, no_decay = mixer_param_groups(m, 1e-4)
    qk = {n for n in names.values() if ".q_norm." in n or ".k_norm." in n}
    assert len(qk) == 2 * len(blocks) and qk <= {names[id(p)] for p in no_decay["params"]}
    with torch.no_grad():
        s = m.representation(torch.rand(2, 12, 96, 96))
    assert torch.isfinite(s).all()
    assert not any(".q_norm." in n for n, _ in EZV2Model(4, obs_channels=12, trunk="qwen", mixer_size="small").named_parameters())


def test_atari_qk_norm_needs_the_qwen_block():
    with pytest.raises(AssertionError):
        EZV2Model(4, obs_channels=12, trunk="gtrxl", mixer_qk_norm=True)
