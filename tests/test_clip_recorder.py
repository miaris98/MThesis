import json

import numpy as np

from src.eval.clip_recorder import ClipRecorder


def test_ring_buffer_clip_has_before_and_after_frames_and_the_event(tmp_path):
    rec = ClipRecorder(str(tmp_path), "route1", hz=4.0, before_s=2.0, after_s=1.0, tick_dt=0.05)
    img = np.zeros((64, 128, 3), np.uint8)
    for i in range(200):                                  # 10 s at 20 Hz
        t = i * 0.05
        rec.push(t, img, {"speed": float(i), "probs": [0.5, 0.5]})
        if i == 100:
            assert rec.trigger(t, "vehicle", {"other_type": "vehicle.audi.a2", "ego_speed": 6.0})
        if i == 101:
            assert not rec.trigger(t, "vehicle", {})      # rate limit: same actor repeat inside 5 s
    rec.close()
    assert len(rec.written) == 1
    z = np.load(rec.written[0], allow_pickle=True)
    ev = json.loads(str(z["event"]))
    sc = json.loads(str(z["scalars"]))
    assert ev["kind"] == "vehicle" and ev["other_type"] == "vehicle.audi.a2"
    assert int(z["n_before"]) == 8 and 8 < len(sc) <= 8 + 5   # 2 s before at 4 Hz, ~1 s after
    assert len(z["jpegs"]) == len(sc) and z["jpegs"][0][:2] == b"\xff\xd8"
    assert sc[int(z["n_before"]) - 1]["t"] <= ev["t"] <= sc[int(z["n_before"])]["t"] + 0.3


def test_clip_is_written_on_close_when_the_route_ends_inside_the_after_window(tmp_path):
    rec = ClipRecorder(str(tmp_path), "r", after_s=5.0)
    img = np.zeros((32, 64, 3), np.uint8)
    for i in range(60):
        rec.push(i * 0.05, img, {})
    rec.trigger(2.9, "stuck", {})
    rec.close()
    assert len(rec.written) == 1
