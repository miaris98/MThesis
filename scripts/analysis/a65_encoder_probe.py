"""TODO A65 (2026-10-14): is a CARLA-fine-tuned InternVL2-1B vision tower (SimLingo, epoch 13) a better frozen perception stage than our regnety_032 (TF++, arm E15)?

Both encoders see the same PDM-Lite frames. Features are pooled to a 4x4 token grid (our head's grid), reduced with PCA fitted on the training routes, and read out by a
linear model on held-out ROUTES (route-level split, no frame leakage). Targets come from the PDM-Lite measurements: the expert's target speed, "must slow down" (speed - target
speed > 1 m/s), "an object caused the slowdown" and its distance. Reports the metric of each encoder and a paired bootstrap of the difference (SimLingo minus regnety).

    python scripts/analysis/a65_encoder_probe.py inspect --simlingo /workspace/simlingo/pytorch_model.pt
    python scripts/analysis/a65_encoder_probe.py run --data_dir /workspace/dataset/wor_trajectories --frozen_backbone <E15 frozen_backbone.pth> \
        --simlingo /workspace/simlingo/pytorch_model.pt --internvl /workspace/internvl2_1b --n_frames 6000 --out /workspace/probe/a65_result.json

The SimLingo preprocessing is theirs (simlingo_training/dataloader/dataset_base.py, utils/internvl2_utils.py): drop the bottom 4.8/16 of the 1024x512 image (the bonnet), cut it into
448x448 tiles with InternVL's dynamic aspect-ratio rule (max 2 tiles), ImageNet normalisation. Mirroring the checkpoint is a download of an untrusted file: it is read with
torch.load(weights_only=True) and nothing from the SimLingo repository is executed; the InternVL2-1B modelling code comes from the OpenGVLab snapshot.
Logs to MLflow (file store) as CLAUDE.md requires.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---- SimLingo / InternVL preprocessing (copied in spirit from simlingo_training/utils/internvl2_utils.py) ----------------------------------------------------------
def _closest_ratio(aspect, ratios, w, h, size):
    best, best_diff = (1, 1), float("inf")
    for r in ratios:
        d = abs(aspect - r[0] / r[1])
        if d < best_diff:
            best, best_diff = r, d
        elif d == best_diff and w * h > 0.5 * size * size * r[0] * r[1]:
            best = r
    return best


def dynamic_tiles(img, size=448, max_num=2):
    from PIL import Image
    w, h = img.size
    ratios = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    rw, rh = _closest_ratio(w / h, ratios, w, h, size)
    img = img.resize((size * rw, size * rh))
    return [img.crop(((i % rw) * size, (i // rw) * size, (i % rw + 1) * size, (i // rw + 1) * size)) for i in range(rw * rh)], (rh, rw)


def simlingo_pixels(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.crop((0, 0, w, h - int(h * 4.8 // 16)))
    tiles, grid = dynamic_tiles(im)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    px = torch.stack([(torch.from_numpy(np.asarray(t.resize((448, 448), Image.BICUBIC))).permute(2, 0, 1).float() / 255 - mean) / std for t in tiles])
    return px, grid


# ---- loading the checkpoints --------------------------------------------------------------------------------------------------------------------------------------
def load_simlingo_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("state_dict", "module", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return sd


def cmd_inspect(a):
    sd = load_simlingo_state(a.simlingo)
    print("tensors:", len(sd), "total params (M):", round(sum(v.numel() for v in sd.values()) / 1e6, 1))
    groups = {}
    for k, v in sd.items():
        g = ".".join(k.split(".")[:4])
        groups[g] = groups.get(g, 0) + v.numel()
    for g, n in sorted(groups.items(), key=lambda x: -x[1])[:25]:
        print(f"{n / 1e6:9.2f} M  {g}")
    print("sha256", hashlib.sha256(open(a.simlingo, "rb").read()).hexdigest())


def build_simlingo(a, device):
    from transformers import AutoModel
    m = AutoModel.from_pretrained(a.internvl, trust_remote_code=True, torch_dtype=torch.bfloat16)
    m.language_model = None
    sd = load_simlingo_state(a.simlingo)
    pre = [k for k in sd if k.endswith("vision_model.embeddings.class_embedding")]
    assert pre, "no InternViT class_embedding key found: run `inspect` and fix the prefix"
    prefix = pre[0][: -len("vision_model.embeddings.class_embedding")]          # e.g. 'vision_model.image_encoder.model.'
    own = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix) and (k[len(prefix):].startswith("vision_model.") or k[len(prefix):].startswith("mlp1."))}
    missing, unexpected = m.load_state_dict({k: v.to(torch.bfloat16) for k, v in own.items()}, strict=False)
    miss = [k for k in missing if k.startswith("vision_model.") or k.startswith("mlp1.")]
    print(f"SimLingo weights: prefix '{prefix}', {len(own)} tensors loaded, missing vision/mlp1 {len(miss)}, unexpected {len(unexpected)}", flush=True)
    assert not miss, f"vision tower not fully covered: {miss[:5]}"
    return m.to(device).eval()


def build_regnety(a, device):
    from src.models.world_on_rails.vision_encoder import build_vision_encoder
    enc = build_vision_encoder(backbone_name="regnety_032", pretrained=False, freeze_backbone=True, weights_path=None)
    sd = torch.load(a.frozen_backbone, map_location="cpu")
    sd = sd.get("model", sd)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, _ = enc.load_state_dict(enc_sd, strict=False)
    assert enc_sd and not missing, f"frozen_backbone.pth does not hold this encoder: missing {list(missing)[:5]}"
    return enc.to(device).eval()


# ---- frames and labels --------------------------------------------------------------------------------------------------------------------------------------------
def meas_path(rgb_path):
    d, f = os.path.split(rgb_path)
    return os.path.join(os.path.dirname(d), "measurements", os.path.splitext(f)[0] + ".json.gz")


def labels(rgb_path):
    with gzip.open(meas_path(rgb_path), "rt") as fh:
        m = json.load(fh)
    speed, tgt = float(m.get("speed", 0.0)), float(m.get("target_speed", m.get("speed", 0.0)))
    dist = m.get("speed_reduced_by_obj_distance")
    return dict(speed=speed, target_speed=tgt, slow=float(speed - tgt > 1.0), obj=float(dist is not None), obj_dist=float(min(dist, 60.0)) if dist is not None else float("nan"))


def route_of(p):
    return os.path.dirname(os.path.dirname(p))


def cmd_run(a):
    from src.training.wor_dataset import WorldOnRailsDataset
    dev = torch.device("cuda")
    ds = WorldOnRailsDataset(data_dir=a.data_dir, is_train=True, cache_decoded=False, route_points=4, img_size=(288, 768), crop_bottom_frac=0.0, route_overlay=True,
                             use_augmented_camera=False)
    S = [s for s in ds.samples if s.get("format") == "pdm_lite" and s.get("rgb_path")]
    rng = np.random.default_rng(0)
    base = rng.choice(len(S), size=min(a.n_frames, len(S)), replace=False)
    lab_cache = {}
    def lab(i):
        if i not in lab_cache:
            lab_cache[i] = labels(S[i]["rgb_path"])
        return lab_cache[i]
    pool = set(base.tolist())
    extra = [i for i in rng.permutation(len(S))[: a.n_frames * 3] if i not in pool and lab(int(i))["slow"] > 0][: a.n_frames // 3]   # oversample "must slow down" frames
    idx = sorted(pool | set(int(i) for i in extra))
    print(f"{len(S)} frames indexed, probing {len(idx)} ({len(extra)} added from slow-down frames)", flush=True)
    Y = {k: np.array([lab(i)[k] for i in idx], np.float32) for k in ("target_speed", "slow", "obj", "obj_dist")}
    routes = np.array([route_of(S[i]["rgb_path"]) for i in idx])
    test_routes = {r for r in set(routes) if int(hashlib.md5(r.encode()).hexdigest(), 16) % 10 < 3}
    test = np.array([r in test_routes for r in routes])
    print(f"routes {len(set(routes))}, test routes {len(test_routes)}, train/test frames {int((~test).sum())}/{int(test.sum())}", flush=True)

    t0 = time.time()
    reg = build_regnety(a, dev)
    fa = []
    with torch.no_grad():
        for b in range(0, len(idx), 64):
            x = torch.stack([torch.as_tensor(np.ascontiguousarray(ds._load_rgb(S[i]["rgb_path"], route_xy=S[i].get("route"))), dtype=torch.uint8) for i in idx[b:b + 64]])
            x = x.to(dev).permute(0, 3, 1, 2).float().div_(255.0)
            fa.append(F.adaptive_avg_pool2d(reg(x).float(), (4, 4)).flatten(1).cpu())
    FA = torch.cat(fa).numpy()
    del reg
    print(f"regnety features {FA.shape} in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    sl = build_simlingo(a, dev)
    fb = []
    with torch.no_grad():
        for i in idx:
            px, (rh, rw) = simlingo_pixels(S[i]["rgb_path"])
            tok = sl.extract_feature(px.to(dev, torch.bfloat16)).float()                                  # (tiles, 256, C)
            t, n, c = tok.shape
            g = tok.view(rh, rw, 16, 16, c).permute(4, 0, 2, 1, 3).reshape(1, c, rh * 16, rw * 16)       # tiles -> one (C, 16*rh, 16*rw) grid
            fb.append(F.adaptive_avg_pool2d(g, (4, 4)).flatten(1).cpu())
    FB = torch.cat(fb).numpy()
    del sl
    print(f"simlingo features {FB.shape} in {time.time() - t0:.0f}s", flush=True)

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def reduce(Fm):
        sc = StandardScaler().fit(Fm[~test])
        Z = sc.transform(Fm)
        p = PCA(n_components=256, random_state=0).fit(Z[~test])
        return p.transform(Z)

    ZA, ZB = reduce(FA), reduce(FB)
    out, mlflow_metrics = {}, {}
    rb = np.random.default_rng(1)
    for name, kind, mask in (("target_speed", "reg", np.ones(len(idx), bool)), ("slow", "clf", np.ones(len(idx), bool)), ("obj", "clf", np.ones(len(idx), bool)),
                             ("obj_dist", "reg", ~np.isnan(Y["obj_dist"]))):
        tr, te = (~test) & mask, test & mask
        if te.sum() < 50 or tr.sum() < 200 or (kind == "clf" and len(set(Y[name][te])) < 2):
            out[name] = "too few frames"
            continue
        preds = {}
        for tag, Z in (("regnety", ZA), ("simlingo", ZB)):
            if kind == "reg":
                m = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Z[tr], Y[name][tr])
                preds[tag] = m.predict(Z[te])
            else:
                m = LogisticRegression(C=0.05, max_iter=500).fit(Z[tr], Y[name][tr])
                preds[tag] = m.predict_proba(Z[te])[:, 1]
        y = Y[name][te]
        score = (lambda p, yy: -np.abs(p - yy).mean()) if kind == "reg" else (lambda p, yy: roc_auc_score(yy, p))
        s = {t: float(score(p, y)) for t, p in preds.items()}
        diffs = []
        for _ in range(500):
            j = rb.integers(0, len(y), len(y))
            if kind == "clf" and len(set(y[j])) < 2:
                continue
            diffs.append(score(preds["simlingo"][j], y[j]) - score(preds["regnety"][j], y[j]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[name] = dict(kind=kind, metric="neg_MAE" if kind == "reg" else "AUROC", regnety=s["regnety"], simlingo=s["simlingo"], diff=s["simlingo"] - s["regnety"], ci=[float(lo), float(hi)], n_test=int(te.sum()))
        print(name, json.dumps(out[name]), flush=True)
        mlflow_metrics.update({f"{name}/regnety": s["regnety"], f"{name}/simlingo": s["simlingo"]})
    out["meta"] = dict(n_frames=len(idx), n_test_routes=len(test_routes), dims=dict(regnety=int(FA.shape[1]), simlingo=int(FB.shape[1])), pca=256)
    json.dump(out, open(a.out, "w"), indent=1)
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow
        mlflow.set_tracking_uri("file:///workspace/mlruns_probe")
        mlflow.set_experiment("A65_encoder_probe")
        with mlflow.start_run(run_name="regnety_vs_simlingo"):
            mlflow.log_params(dict(n_frames=len(idx), pca=256, grid="4x4"))
            mlflow.log_metrics(mlflow_metrics)
            mlflow.log_artifact(a.out)
    except Exception as ex:
        print("mlflow logging skipped:", ex)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("inspect")
    i.add_argument("--simlingo", required=True)
    r = sp.add_parser("run")
    r.add_argument("--data_dir", required=True)
    r.add_argument("--frozen_backbone", required=True)
    r.add_argument("--simlingo", required=True)
    r.add_argument("--internvl", required=True)
    r.add_argument("--n_frames", type=int, default=6000)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_inspect(a) if a.cmd == "inspect" else cmd_run(a)
