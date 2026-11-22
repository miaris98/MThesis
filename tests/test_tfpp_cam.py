"""TODO A86 (2026-11-21): TF++ camera-only on our input. What must hold before a box is rented:

1. TF++'s camera-only encoder, loaded from E15's frozen_backbone.pth, gives the same stride-32 map as our encoder on the same image
   (so the cache our policy is trained from is exactly what TF++'s decoder would see live), and the map is 9x24 at 288x768;
2. the model trains on a cached map through the pass-through backbone, and only the decoder / heads get gradients (encoder frozen);
3. the labels are TF++'s: shapes, a straight route stays straight, the shifted-camera rule moves the route and the target point;
4. the saved directory loads into a fresh LidarCenterNet(config) with strict=True and reproduces the outputs.
"""
import os

import numpy as np
import pytest
import torch

from src.baselines import tfpp_cam as tc
from tests._tfpp_isolation import isolate_tfpp_imports

_isolate = isolate_tfpp_imports()   # carla / shapely / agents stand-ins and team_code must not leak into later test modules

FB = "E:/MThesis_EXP/live_20261006_box0_train_3090ti/checkpoints/carla_armE_aug1_hires/frozen_backbone.pth"
need_fb = pytest.mark.skipif(not os.path.exists(FB), reason="E15's frozen backbone (E:) not available")


@pytest.fixture(scope="module")
def cfg():
    return tc.make_config((288, 768))


@pytest.fixture(scope="module")
def net(cfg):
    torch.manual_seed(0)
    return tc.build_model(cfg, None)


def test_config_is_camera_only_on_our_geometry(cfg):
    assert cfg.backbone == "aim" and not cfg.crop_image and (cfg.img_vert_anchors, cfg.img_horz_anchors) == (9, 24)
    assert not (cfg.use_semantic or cfg.use_depth or cfg.use_bev_semantic or cfg.detect_boxes or cfg.use_color_aug)
    assert cfg.use_controller_input_prediction and cfg.transformer_decoder_join and cfg.epochs == 31


@need_fb
def test_encoder_parity_with_our_encoder(cfg):
    from src.models.world_on_rails.carla_encoder import CarlaPretrainedEncoder
    state = torch.load(FB, map_location="cpu", weights_only=False)
    tf = tc.build_model(cfg, state).eval()
    ours = CarlaPretrainedEncoder(expect_external_weights=True).eval()       # the evaluation path: weights from frozen_backbone.pth
    flat = state["model"] if isinstance(state, dict) and isinstance(state.get("model"), dict) else state
    ours.load_state_dict({k[len("encoder."):]: v for k, v in flat.items() if k.startswith("encoder.")}, strict=True)
    img = torch.rand(2, 3, 288, 768) * 255.0                                  # 0-255, as both pipelines receive it
    with torch.no_grad():
        a, _ = tf.backbone(img)                      # TF++: normalize_imagenet inside, then the 4 stages
        b = ours(img)                                # ours: /255, ImageNet mean/std, the same stages
    assert a.shape == (2, 1512, 9, 24)
    assert b.shape == a.shape
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-3), float((a - b).abs().max())


def _batch(b=4):
    g = torch.Generator().manual_seed(2)
    return {"map": torch.randn(b, 1512, 9, 24, generator=g), "target_point": torch.randn(b, 2, generator=g) * 10,
            "speed": torch.rand(b, generator=g) * 8, "command": torch.eye(6)[torch.tensor([3, 3, 0, 1])[:b]],
            "checkpoint": torch.cumsum(torch.rand(b, 10, 2, generator=g), 1), "target_speed_twohot": torch.softmax(torch.randn(b, 8, generator=g), -1),
            "target_speed_index": torch.randint(0, 8, (b,), generator=g)}


def _forward(net, batch):
    return net(rgb=batch["map"], lidar_bev=None, target_point=batch["target_point"], ego_vel=batch["speed"].unsqueeze(1),
               command=batch["command"])


def test_trains_on_cached_maps_with_the_encoder_frozen(cfg):
    torch.manual_seed(0)
    net = tc.build_model(cfg, None)
    tc.use_cached_maps(net)
    net.train()
    batch = _batch()
    out = _forward(net, batch)
    assert out[2].shape == (4, 10, 2) and out[1].shape == (4, len(cfg.target_speeds))
    loss = tc.planning_loss(net, out, batch)
    assert set(loss) >= {"total", "loss_checkpoint", "loss_target_speed"} and torch.isfinite(loss["total"])
    loss["total"].backward()
    assert all(p.grad is None for p in net.backbone.image_encoder.parameters())
    assert net.change_channel.weight.grad is not None and net.change_channel.weight.grad.abs().sum() > 0


def _meas(n=30):
    return {"route": [[float(i), 0.0] for i in range(1, n + 1)], "target_point": [20.0, 0.0], "target_speed": 7.5, "brake": False,
            "angle": 0.0, "speed": 6.0, "command": 4, "augmentation_rotation": 10.0, "augmentation_translation": 1.0}


def test_labels_are_tfpp_shapes_and_the_shift_rule_moves_them(cfg):
    lab = tc.labels_for(cfg)
    plain, shifted = lab(_meas(), augmented=False), lab(_meas(), augmented=True)
    assert plain["checkpoint"].shape == (cfg.predict_checkpoint_len, 2) and plain["target_point"].shape == (2,)
    assert np.allclose(plain["checkpoint"][:, 1], 0.0, atol=1e-6)                  # a straight route stays straight
    assert plain["target_speed_twohot"].shape == (len(cfg.target_speeds),) and abs(plain["target_speed_twohot"].sum() - 1) < 1e-5
    assert plain["command"].shape == (6,) and plain["command"].sum() == 1
    assert not np.allclose(shifted["checkpoint"], plain["checkpoint"]) and not np.allclose(shifted["target_point"], plain["target_point"])
    braking = lab({**_meas(), "brake": True}, augmented=False)
    assert braking["target_speed_index"] == 0


def test_agent_image_path_equals_our_preprocessing(cfg):
    """The agent's crop_array hook: a camera-only config gets OUR resize (identical array to src.config.camera.preprocess_rgb); any other
    config, or a 2-D depth/semantic array, keeps TF++'s own crop."""
    import transfuser_utils as t_u
    from src.config.camera import preprocess_rgb
    hook = tc.crop_or_resize_like_ours(t_u.crop_array)
    img = np.random.RandomState(0).randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    out = hook(cfg, img)
    assert out.shape == (288, 768, 3) and np.array_equal(out, preprocess_rgb(img, img_size=(288, 768), crop_bottom_frac=0.0))
    other = tc.make_config((288, 768))
    del other.tfpp_cam_img_hw
    other.crop_image = True
    assert hook(other, img).shape == (other.cropped_height, other.cropped_width, 3)    # TF++'s crop untouched
    assert hook(cfg, img[..., 0]).shape == (512, 1024)                                   # 2-D arrays untouched


def test_saved_dir_loads_strict_and_reproduces(cfg, tmp_path):
    torch.manual_seed(0)
    net = tc.build_model(cfg, None).eval()
    path = tc.save_tfpp_dir(net, cfg, str(tmp_path / "tfpp_cam"), 31)
    import jsonpickle
    loaded_cfg = jsonpickle.decode(open(tmp_path / "tfpp_cam" / "config.json", encoding="utf-8").read())
    assert loaded_cfg.backbone == "aim" and loaded_cfg.img_horz_anchors == 24
    fresh = tc.build_model(loaded_cfg, None)
    fresh.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    fresh.eval()
    img = torch.rand(1, 3, 288, 768) * 255.0
    b = _batch(1)
    with torch.no_grad():
        o1 = net(rgb=img, lidar_bev=None, target_point=b["target_point"], ego_vel=b["speed"].unsqueeze(1), command=b["command"])
        o2 = fresh(rgb=img, lidar_bev=None, target_point=b["target_point"], ego_vel=b["speed"].unsqueeze(1), command=b["command"])
    assert torch.allclose(o1[2], o2[2]) and torch.allclose(o1[1], o2[1])
