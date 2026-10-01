"""TODO A16 (arm K): swerve frames are marked at index time and oversampled to a target share by the train loader."""
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from src.training.swerve import route_shift as _route_shift, swerve_sampler
from src.training.wor_dataset import WorldOnRailsDataset


def _route(root: Path, n_frames: int, shifted=(), shift=1.5):
    """A straight PDM-Lite route; at the frames in `shifted` the expert's `route` is bent `shift` m left of the plan."""
    from PIL import Image
    route = root / "Town12" / "route_0"
    (route / "measurements").mkdir(parents=True)
    (route / "rgb").mkdir()
    plan = [[float(k), 0.0] for k in range(20)]
    for i in range(n_frames):
        m = np.eye(4); m[0, 3] = float(i)
        bent = [[x, shift if i in shifted else 0.0] for x, _ in plan]
        meas = {"speed": 5.0, "command": 4, "ego_matrix": m.tolist(), "route": bent, "route_original": plan,
                "target_speed": 5.0}
        with gzip.open(route / "measurements" / f"{i:04d}.json.gz", "wt") as f:
            json.dump(meas, f)
        Image.fromarray(np.full((32, 64, 3), 128, np.uint8)).save(route / "rgb" / f"{i:04d}.jpg")


def test_route_shift():
    plan = [[float(k), 0.0] for k in range(10)]
    assert _route_shift(plan, plan) == 0.0
    assert _route_shift([[x, 0.8] for x, _ in plan], plan) == 0.8
    assert _route_shift(plan, None) == 0.0 and _route_shift([], plan) == 0.0
    # only the first 4 points (what the policy sees) count
    late = [[x, 3.0 if k > 5 else 0.0] for k, (x, _) in enumerate(plan)]
    assert _route_shift(late, plan) == 0.0


def test_swerve_frames_include_the_approach_window(tmp_path, monkeypatch):
    monkeypatch.setenv("WOR_SWERVE_WINDOW", "3")
    _route(tmp_path, n_frames=30, shifted=(15,))
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=False)
    frame = lambda s: int(Path(s["rgb_path"]).stem)
    flagged = sorted(frame(s) for s in ds.samples if s["swerve"])
    assert flagged == [12, 13, 14, 15]          # the shifted frame and the 3 before it, not after


def test_small_shift_is_not_a_swerve(tmp_path):
    _route(tmp_path, n_frames=20, shifted=(10,), shift=0.3)
    ds = WorldOnRailsDataset(data_dir=str(tmp_path), img_size=(32, 64), cache_decoded=False)
    assert not any(s["swerve"] for s in ds.samples)


class _Fake:
    def __init__(self, flags):
        self.samples = [{"swerve": f} for f in flags]

    def __len__(self):
        return len(self.samples)


def test_sampler_hits_the_target_share_and_keeps_epoch_length():
    flags = [True] * 20 + [False] * 980                    # 2% natural share
    s = swerve_sampler(_Fake(flags), 0.10, seed=0)
    assert s.num_samples == 1000
    draws = torch.tensor(list(iter(s)))
    torch.manual_seed(0)
    share = np.mean([flags[i] for i in torch.multinomial(s.weights, 200_000, replacement=True).tolist()])
    assert abs(share - 0.10) < 0.005
    assert len(draws) == 1000


def test_sampler_never_lowers_a_natural_share_above_target():
    s = swerve_sampler(_Fake([True] * 300 + [False] * 700), 0.10, seed=0)
    assert torch.all(s.weights == 1.0)


def test_sampler_on_a_subset_uses_the_subset_indices():
    base = _Fake([True] * 10 + [False] * 90)
    sub = Subset(base, list(range(5, 100)))                 # 5 swerve frames left out of 95
    s = swerve_sampler(sub, 0.5, seed=0)
    assert s.num_samples == 95
    assert abs(float(s.weights[:5].sum() / s.weights.sum()) - 0.5) < 1e-9


def test_sampler_is_seeded():
    flags = [True] * 20 + [False] * 180
    a = list(iter(swerve_sampler(_Fake(flags), 0.1, seed=3)))
    b = list(iter(swerve_sampler(_Fake(flags), 0.1, seed=3)))
    assert a == b
