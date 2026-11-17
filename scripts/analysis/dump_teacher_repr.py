"""Writes the privileged teacher's policy token for every row of a pooled cache (2026-11-08, idea 8).

The teacher (train_wor.py --privileged_boxes M) reads the M nearest logged objects instead of the camera. Its post-trunk policy token per
frame is written as <out> (N, 512) fp16 (+ <out>.json with the shape), rows aligned with the cache's keys, so a camera student trained with
--teacher_repr <out> --distill_weight w learns to put into its own token what the teacher knows. Every cache row is written (train and
held-out frames alike; the held-out frames are only used for the student's validation loss, which does not include the distillation term).

    python scripts/analysis/dump_teacher_repr.py --teacher <ckpt dir>/model_epoch_030.pth --prefix <pooled cache prefix> --data_dir <dataset> \\
        --boxes 20 --weights_path <TF++ model_0030_0.pth> --out <prefix>.teacher20.f16
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--boxes", type=int, default=20)
    ap.add_argument("--weights_path", required=True)
    ap.add_argument("--img_size", default="288x768")
    ap.add_argument("--crop_bottom_frac", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.environ["WOR_POOLED_CACHE"] = a.prefix
    os.environ["WOR_PRIVILEGED"] = str(a.boxes)
    os.environ["WOR_ROUTE_KEY"] = "route_original"
    from src.models.world_on_rails.qwen_wor_policy import QwenWorldOnRailsPolicy
    from src.training.pooled_feature_cache import rel_key
    from src.training.wor_dataset import WorldOnRailsDataset
    from src.training.wor_eval import to_device_batch_with_features

    h, w = (int(v) for v in a.img_size.split("x"))
    # is_train=True: the training index holds the recovery-camera frames as their own samples (the cache's rows); no augmentation is drawn
    # (colour augmentation off by default, no flip cache), so every row's input is the exact one the student sees.
    ds = WorldOnRailsDataset(data_dir=a.data_dir, is_train=True, cache_decoded=False, route_points=4, img_size=(h, w),
                             crop_bottom_frac=a.crop_bottom_frac, route_overlay=True, use_augmented_camera=True)
    model = QwenWorldOnRailsPolicy(backbone_name="regnety_032", pretrained=True, freeze_backbone=True, weights_path=a.weights_path, route_points=4,
                                   model_size="30m", vision_grid=4, use_target_speed=True, use_rail_q=False, privileged_boxes=a.boxes,
                                   crop_bottom_frac=a.crop_bottom_frac)
    ck = torch.load(a.teacher, map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("encoder.")]
    assert not missing and not unexpected, f"teacher checkpoint does not fit: missing {missing[:5]}, unexpected {list(unexpected)[:5]}"
    model = model.to(a.device).eval()
    meta = json.load(open(a.prefix + ".json"))
    N = meta["n"]
    D = model.embed_dim
    out = np.memmap(a.out, dtype=np.float16, mode="w+", shape=(N, D))
    done = np.zeros(N, dtype=bool)
    rows = np.array([ds._pooled.row_of(rel_key(s["rgb_path"], a.data_dir))[1] for s in ds.samples])
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False)
    i0 = 0
    with torch.no_grad():
        for batch in loader:
            rgb, vf, speed, command, route, _, _ = to_device_batch_with_features(batch, a.device)
            model(rgb, speed, command, route, vision_features=vf)
            rep = model._last_policy_repr.float().cpu().numpy()
            r = rows[i0:i0 + len(rep)]
            out[r] = rep.astype(np.float16)
            done[r] = True
            i0 += len(rep)
    out.flush()
    json.dump({"shape": [N, D], "teacher": a.teacher, "boxes": a.boxes, "rows_written": int(done.sum())}, open(a.out + ".json", "w"))
    print(f"--> teacher tokens: {int(done.sum())} of {N} rows -> {a.out}")
    if int(done.sum()) != N:
        raise SystemExit(f"only {int(done.sum())} of {N} cache rows got a teacher token: the dataset index does not cover the cache")
    print("TEACHER_REPR_DONE")


if __name__ == "__main__":
    main()
