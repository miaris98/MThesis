"""2026-11-21 memorization test (stage 1): training-only regularizers on the camera tokens and the clean training-error readout.

1. feature_noise / token_drop never act in eval mode: driving and validation see the model exactly as without them;
2. in training they do act, on the camera-grid tokens only (R1's object tokens after the grid are untouched), at the asked strength;
3. they add no tensors, so a checkpoint loads into a model built without them;
4. layouts where the grid is not the first gh*gw tokens, and bad values, are refused at construction and on the command line;
5. the clean training readout samples training frames only, main camera only, the same frames every time, and refuses augmentation;
6. a short real training run writes the train_clean_* columns, scored in eval mode (two calls agree exactly).
"""
import argparse
import csv
import math

import pytest
import torch
from torch.utils.data import Subset

from src.models.world_on_rails import QwenWorldOnRailsPolicy
import src.training.wor_dataset  # noqa: F401  (first: wor_dataset and wor_dataloaders import each other; the trainer does the same)
from src.training.wor_dataloaders import _wrap_loader, make_train_eval_loader, route_group_split


def _model(**kw):
    torch.manual_seed(0)
    return QwenWorldOnRailsPolicy(backbone_name="resnet18", pretrained=False, route_points=4, model_size="10m",
                                  vision_grid=4, use_target_speed=True, **kw)


def _batch(b=4):
    g = torch.Generator().manual_seed(1)
    return (torch.rand(b, 3, 128, 128, generator=g), torch.tensor([0.0, 2.0, 6.0, 12.0])[:b],
            torch.tensor([3, 3, 2, 4])[:b], torch.rand(b, 4, 2, generator=g))


def test_regularizers_are_off_in_eval_mode():
    plain = _model().eval()
    reg = _model(feature_noise=0.5, token_drop=0.3).eval()
    reg.load_state_dict(plain.state_dict())
    with torch.no_grad():
        a, b = plain(*_batch()), reg(*_batch())
    assert torch.equal(a["waypoints"], b["waypoints"])
    assert torch.equal(a["target_speed_logits"], b["target_speed_logits"])


def test_regularizers_act_in_training_and_are_seeded():
    reg = _model(feature_noise=0.5, token_drop=0.3).train()
    plain = _model().train()
    plain.load_state_dict(reg.state_dict())
    torch.manual_seed(5)
    out1 = reg(*_batch())["waypoints"]
    torch.manual_seed(5)
    out2 = reg(*_batch())["waypoints"]
    torch.manual_seed(5)
    ref = plain(*_batch())["waypoints"]
    assert torch.equal(out1, out2), "same seed, same noise"
    assert not torch.allclose(out1, ref), "the regularized model must differ from the plain one in training"


def test_token_drop_rate_and_grid_only():
    m = _model(token_drop=0.3).train()
    tok = torch.ones(4000, 16 + 3, 8)          # 16 grid tokens, then 3 tokens appended after the grid (R1's object tokens)
    torch.manual_seed(0)
    out = m._regularize_grid(tok)
    dropped = (out[:, :16].abs().sum(-1) == 0).float().mean().item()
    assert dropped == pytest.approx(0.3, abs=0.01)
    assert torch.equal(out[:, 16:], tok[:, 16:]), "tokens after the grid are never touched"
    assert torch.equal(out[:, :16][out[:, :16] != 0], torch.ones_like(out[:, :16][out[:, :16] != 0])), "kept tokens are not rescaled"


def test_feature_noise_scale_is_relative_to_each_token():
    m = _model(feature_noise=0.25).train()
    tok = torch.randn(512, 16, 256) * torch.linspace(0.1, 10.0, 16).view(1, 16, 1)   # tokens of very different size
    torch.manual_seed(0)
    out = m._regularize_grid(tok)
    rel = (out - tok).pow(2).mean(dim=(0, 2)).sqrt() / tok.pow(2).mean(dim=(0, 2)).sqrt()
    assert torch.allclose(rel, torch.full_like(rel, 0.25), rtol=0.05), rel


def test_feature_noise_does_not_backprop_through_its_scale():
    m = _model(feature_noise=0.5).train()
    tok = torch.randn(8, 16, 32, requires_grad=True)
    torch.manual_seed(0)
    m._regularize_grid(tok).sum().backward()
    assert torch.allclose(tok.grad, torch.ones_like(tok.grad)), "noise scale is detached: d(out)/d(in) = 1"


def test_regularizers_add_no_tensors():
    assert set(_model(feature_noise=0.5, token_drop=0.3).state_dict()) == set(_model().state_dict())


@pytest.mark.parametrize("kw", [dict(feature_noise=-0.1), dict(token_drop=1.0), dict(token_drop=-0.1),
                                dict(feature_noise=0.1, vision_grid=0), dict(token_drop=0.1, route_windows=4)])
def test_bad_values_and_layouts_are_refused(kw):
    base = dict(backbone_name="resnet18", pretrained=False, route_points=4, model_size="10m", vision_grid=4, use_target_speed=True)
    base.update(kw)
    with pytest.raises(ValueError):
        QwenWorldOnRailsPolicy(**base)


def _args(**kw):
    a = dict(feature_noise=0.0, token_drop=0.0, policy_arch="qwen30m", route_windows=0, attn_pool=0, query_decoder=0,
             privileged_boxes=0, token_mode="pool", train_eval_frames=0, val_data_dir=None, val_split=0.15, num_folds=1,
             flip_prob=0.0, color_aug_prob=0.0, weight_decay=1e-4)
    a.update(kw)
    return argparse.Namespace(**a)


def test_weight_decay_reaches_the_decayed_groups_only(tmp_path):
    """A87 wd arm: --weight_decay sets the decayed groups; gates, norms and learned tokens stay at 0."""
    from src.training.wor_trainer import WorldOnRailsTrainer
    t = WorldOnRailsTrainer(model=_model(), data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"), batch_size=4, num_workers=0,
                            device="cpu", synthetic_samples=8, use_mlflow=False, weight_decay=0.01)
    wds = sorted({g["weight_decay"] for g in t.optimizer.param_groups if g["params"]})
    assert wds == [0.0, 0.01]


@pytest.mark.parametrize("kw", [dict(feature_noise=-1), dict(token_drop=1.0), dict(feature_noise=0.1, policy_arch="cnn"),
                                dict(token_drop=0.1, attn_pool=8), dict(token_drop=0.1, token_mode="s2d"),
                                dict(train_eval_frames=-1), dict(train_eval_frames=100, val_split=0.0),
                                dict(train_eval_frames=100, val_data_dir="/x"), dict(train_eval_frames=100, flip_prob=0.3),
                                dict(weight_decay=-0.01)])
def test_command_line_refusals(kw):
    from scripts.training.train_wor import check_regularizer_args
    with pytest.raises(SystemExit):
        check_regularizer_args(_args(**kw))


def test_command_line_accepts_the_ladder_settings():
    from scripts.training.train_wor import check_regularizer_args
    for noise, drop in ((0.0, 0.0), (0.1, 0.1), (0.25, 0.2), (0.5, 0.3)):
        check_regularizer_args(_args(feature_noise=noise, token_drop=drop, train_eval_frames=20000))


class _StubRoutes(torch.utils.data.Dataset):
    """route_group_split's minimum (samples with rgb_path) plus __getitem__, with a recovery re-render per frame."""

    def __init__(self, n_routes=20, frames=10):
        self.samples = []
        for r in range(n_routes):
            for i in range(frames):
                self.samples.append({"rgb_path": f"/d/Route{r:02d}/rgb/{i:04d}.jpg"})
                self.samples.append({"rgb_path": f"/d/Route{r:02d}/rgb_augmented/{i:04d}.jpg", "is_recovery_augmented": True})
        self.color_aug_prob, self._flip_prob = 0.0, 0.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return i


def _loaders(ds):
    tr, va, _, _ = route_group_split(ds, 0.25, split_seed=0)
    return _wrap_loader(Subset(ds, tr), 8, 0, True, 0), tr, va


def test_clean_readout_is_training_main_camera_and_fixed():
    ds = _StubRoutes()
    train_loader, tr, va = _loaders(ds)
    a = make_train_eval_loader(train_loader, 50, seed=0, batch_size=8)
    b = make_train_eval_loader(train_loader, 50, seed=0, batch_size=8)
    picked = a.dataset.indices
    assert len(picked) == 50 and picked == b.dataset.indices, "the same frames every time"
    assert set(picked) <= set(tr) and not set(picked) & set(va), "training side of the split only"
    assert not any(ds.samples[i].get("is_recovery_augmented") for i in picked), "main camera only"
    assert sorted(int(i) for batch in a for i in batch) == picked, "unshuffled, every picked frame once"


def test_clean_readout_caps_at_the_eligible_count_and_refuses_augmentation():
    ds = _StubRoutes(n_routes=4, frames=5)
    train_loader, tr, _ = _loaders(ds)
    eligible = [i for i in tr if not ds.samples[i].get("is_recovery_augmented")]
    assert len(make_train_eval_loader(train_loader, 10_000, seed=0, batch_size=8).dataset) == len(eligible)
    assert make_train_eval_loader(train_loader, 0, seed=0, batch_size=8) is None
    ds._flip_prob = 0.3
    with pytest.raises(ValueError):
        make_train_eval_loader(train_loader, 10, seed=0, batch_size=8)


def test_training_run_writes_the_clean_columns(tmp_path):
    from src.training.wor_trainer import WorldOnRailsTrainer
    trainer = WorldOnRailsTrainer(
        model=_model(feature_noise=0.25, token_drop=0.2), data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"),
        batch_size=4, num_workers=0, device="cpu", synthetic_samples=40, use_mlflow=False, val_split=0.25, seed=0,
        train_eval_frames=8, img_size=(128, 128))
    assert trainer.train_eval_loader is not None and len(trainer.train_eval_loader.dataset) == 8
    trainer.train(num_epochs=1, save_freq=1)

    def scored():   # synthetic frames are drawn at random on every read: fix the draw, so only the model can differ
        import random
        import numpy as np
        random.seed(0), np.random.seed(0), torch.manual_seed(0)
        return trainer.validate_train_clean()

    with_reg = scored()
    trainer.model.feature_noise, trainer.model.token_drop = 0.0, 0.0
    assert scored() == with_reg, "scored in eval mode: the same weights without the regularizers give the same numbers"
    assert trainer.model.training, "run_validation restores train mode"
    with open(tmp_path / "ck" / "wor_training_telemetry.csv", newline="") as f:
        row = list(csv.DictReader(f))[-1]
    for k in ("train_clean_loss", "train_clean_wp_ade_m", "val_wp_ade_m"):
        assert row[k] != "" and math.isfinite(float(row[k])), k


def test_readout_off_leaves_the_columns_blank(tmp_path):
    from src.training.wor_trainer import WorldOnRailsTrainer
    trainer = WorldOnRailsTrainer(
        model=_model(), data_dir=str(tmp_path / "none"), save_dir=str(tmp_path / "ck"), batch_size=4, num_workers=0,
        device="cpu", synthetic_samples=24, use_mlflow=False, val_split=0.25, seed=0, img_size=(128, 128))
    assert trainer.train_eval_loader is None and trainer.validate_train_clean() == {}
    trainer.train(num_epochs=1, save_freq=1)
    with open(tmp_path / "ck" / "wor_training_telemetry.csv", newline="") as f:
        row = list(csv.DictReader(f))[-1]
    assert row["train_clean_wp_ade_m"] == "" and row["val_wp_ade_m"] != ""
