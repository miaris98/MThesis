"""TODO A71 (2026-11-01): training can take the CARLA encoder from a run's own frozen_backbone.pth ({"model": {"encoder.encoder.*"}}), the file
every checkpoint ships with, instead of the TransFuser++ release checkpoint; the strict 492/492 check still guards the load."""
import pytest
import torch

timm = pytest.importorskip("timm")

from src.models.world_on_rails.carla_encoder import CarlaPretrainedEncoder  # noqa: E402


def test_frozen_backbone_file_loads_cleanly(tmp_path):
    src = CarlaPretrainedEncoder(expect_external_weights=True)
    with torch.no_grad():
        for p in src.encoder.parameters():
            p.normal_()
    path = tmp_path / "frozen_backbone.pth"
    torch.save({"model": {f"encoder.encoder.{k}": v for k, v in src.encoder.state_dict().items()}}, path)
    dst = CarlaPretrainedEncoder(weights_path=str(path))                      # strict by default: a partial load raises
    for (k, a), (_, b) in zip(src.encoder.state_dict().items(), dst.encoder.state_dict().items()):
        assert torch.equal(a, b), k
