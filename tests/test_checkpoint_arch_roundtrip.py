"""A checkpoint must rebuild into the architecture it was trained as.

The new flags (`target_speed_input`, `use_rail_q`, `use_ray_geometry`) each change a tensor
shape. `load_state_dict(strict=False)` forgives a missing or unexpected key but still raises on
a size mismatch, so getting this wrong is at least loud - but only if the loader knows what to
rebuild. These tests walk the real path: stamp the config the way WorldOnRailsTrainer does, save,
then load through `load_wor_model` and require a clean load and identical predictions.

Group 11.4 in this project is a catalogue of evaluation quietly running something other than
what was trained. This is that check for the architecture itself.
"""
import pytest
import torch

from src.models.world_on_rails import QwenWorldOnRailsPolicy, WorldOnRailsPolicy
from src.models.world_on_rails.wor_loader import load_wor_model
from src.training.wor_checkpoint import CheckpointWriter

IMG = (192, 512)


def _stamp(model, arch):
    """The subset of WorldOnRailsTrainer.model_config that selects the architecture."""
    return {
        "backbone": model.encoder.backbone_name,
        "route_points": model.route_points,
        "vision_grid": list(getattr(model, "vision_grid_hw", None) or [])
                       or getattr(model, "vision_grid", None),
        "pool_vision": getattr(model, "pool_vision", None),
        "num_vision_tokens": getattr(model, "num_vision_tokens", None),
        "use_target_speed": model.target_speed_head is not None,
        "target_speed_input": model.target_speed_input,
        "use_rail_q": model.use_rail_q,
        "use_ray_geometry": model.use_ray_geometry,
        "crop_bottom_frac": model.crop_bottom_frac,
    }


def _roundtrip(tmp_path, model, arch, capsys):
    path = tmp_path / "ckpt.pth"
    torch.save({"epoch": 1, "model": model.state_dict(), "config": _stamp(model, arch)}, path)
    capsys.readouterr()
    loaded = load_wor_model(
        checkpoint_path=str(path), backbone_name="resnet34", pretrained_backbone=False,
        device="cpu", policy_arch=arch, route_points=4)
    out = capsys.readouterr().out
    assert "Missing keys: 0" in out and "Unexpected keys: 0" in out, out
    return loaded.eval()


def _same_predictions(a, b):
    speed, cmd, route = torch.tensor([[5.0]]), torch.tensor([3]), torch.zeros(1, 4, 2)
    rgb = torch.rand(1, 3, *IMG, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        pa, pb = a(rgb, speed, cmd, route), b(rgb, speed, cmd, route)
    assert set(pa) == set(pb)
    for k in pa:
        assert torch.allclose(pa[k], pb[k], atol=1e-6), k


def test_cnn_roundtrips_with_the_new_flags(tmp_path, capsys):
    m = WorldOnRailsPolicy(
        backbone_name="resnet34", pretrained=False, route_points=4, use_target_speed=True,
        target_speed_input="policy", use_rail_q=False, use_ray_geometry=True).eval()
    loaded = _roundtrip(tmp_path, m, "cnn", capsys)
    assert loaded.target_speed_input == "policy"
    assert loaded.use_rail_q is False and loaded.use_ray_geometry is True
    _same_predictions(m, loaded)


def test_qwen_roundtrips_with_the_new_flags(tmp_path, capsys):
    m = QwenWorldOnRailsPolicy(
        backbone_name="resnet34", pretrained=False, route_points=4, model_size="10m",
        vision_grid=(6, 16), use_target_speed=True, target_speed_input="policy",
        use_rail_q=False, use_ray_geometry=True).eval()
    loaded = _roundtrip(tmp_path, m, "qwen10m", capsys)
    assert loaded.target_speed_input == "policy"
    assert loaded.use_rail_q is False and loaded.use_ray_geometry is True
    _same_predictions(m, loaded)


def test_a_checkpoint_without_the_new_keys_rebuilds_the_old_architecture(tmp_path, capsys):
    """Pre-stamp checkpoints carry none of these keys. They must still load exactly as before -
    state-fed target-speed head, rail heads present, no geometry - rather than silently
    adopting the new defaults and mismatching every affected tensor."""
    old = WorldOnRailsPolicy(
        backbone_name="resnet34", pretrained=False, route_points=4, use_target_speed=True,
        target_speed_input="state", use_rail_q=True, use_ray_geometry=False).eval()
    path = tmp_path / "old.pth"
    torch.save({"epoch": 1, "model": old.state_dict(),
                "config": {"backbone": "resnet34", "route_points": 4,
                           "target_speed_loss_weight": 1.0}}, path)
    capsys.readouterr()
    loaded = load_wor_model(checkpoint_path=str(path), backbone_name="resnet34",
                            pretrained_backbone=False, device="cpu", policy_arch="cnn",
                            route_points=4).eval()
    out = capsys.readouterr().out
    assert "Missing keys: 0" in out and "Unexpected keys: 0" in out, out
    assert loaded.target_speed_input == "state"
    assert loaded.use_rail_q is True and loaded.use_ray_geometry is False
    _same_predictions(old, loaded)


def test_mismatched_architecture_fails_loudly_rather_than_silently(tmp_path):
    """The failure mode worth guarding: a geometry-trained checkpoint rebuilt without geometry.
    strict=False forgives absent keys but not a wrong shape, so this must raise."""
    m = WorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, route_points=4,
                           use_target_speed=True, use_ray_geometry=True)
    path = tmp_path / "geo.pth"
    torch.save({"epoch": 1, "model": m.state_dict(),
                "config": {"backbone": "resnet34", "route_points": 4,
                           "use_ray_geometry": False, "use_target_speed": True}}, path)
    try:
        load_wor_model(checkpoint_path=str(path), backbone_name="resnet34",
                       pretrained_backbone=False, device="cpu", policy_arch="cnn",
                       route_points=4)
    except RuntimeError as exc:
        assert "size mismatch" in str(exc).lower()
    else:
        raise AssertionError("a shape-mismatched checkpoint loaded silently")


def test_rectangular_vision_grid_survives_the_stamp():
    """Regression: the stamp used to record `model.vision_grid`, which collapses a (6, 16)
    grid to the scalar 6. The loader then rebuilt a 6x6 / 36-token trunk and every v2 qwen
    checkpoint - all of which are rectangular - failed to load with a vision_pos size
    mismatch. The stamp must carry both dimensions."""
    m = QwenWorldOnRailsPolicy(
        backbone_name="resnet34", pretrained=False, route_points=4, model_size="10m",
        vision_grid=(6, 16), use_target_speed=False)
    assert m.num_vision_tokens == 96
    assert _stamp(m, "qwen10m")["vision_grid"] == [6, 16]


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile needs torch>=2.0")
def test_frozen_keys_survive_torch_compiles_orig_mod_prefix():
    """Regression, found 2026-09-14: torch.compile's state_dict keys are prefixed
    "_orig_mod.encoder...." instead of "encoder....". CheckpointWriter used to match only the
    unprefixed form, so frozen_keys came back empty for every compiled run - no exception, just
    a silently un-split checkpoint (no frozen_backbone.pth, and every per-epoch save quietly
    ballooned from heads-only to the full model). Caught by comparing checkpoint file sizes
    between a compiled and an uncompiled run of the same architecture."""
    m = WorldOnRailsPolicy(backbone_name="resnet34", pretrained=False, route_points=4,
                           freeze_backbone=True)
    uncompiled_keys = CheckpointWriter(m, "/tmp", freeze_backbone=True).frozen_keys
    assert uncompiled_keys, "sanity: the uncompiled model must find frozen encoder keys at all"

    compiled = torch.compile(m)
    compiled_keys = CheckpointWriter(compiled, "/tmp", freeze_backbone=True).frozen_keys
    assert compiled_keys, (
        "frozen_keys was empty for a compiled model - the _orig_mod. prefix broke the match")
    assert len(compiled_keys) == len(uncompiled_keys)
