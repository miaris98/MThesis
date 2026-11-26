#!/usr/bin/env python3
"""TODO A88: profile one driving policy's training and inference cost on the data it was trained on.

Training runs the run's own code: train_wor.main() with the checkpoint's run_config.json (WoR / Qwen: E15, r0ftf, R1, R1+w4; box paths
swapped for --data-dir / --cache / --frozen-backbone / --detector) with WorldOnRailsTrainer.train() replaced by the measurements, or
train_tfpp_cam.py's dataset, optimizer and step (A86). Inference runs the deployed agent (WorldOnRailsAgent.run_step / TF++ net +
control_pid_direct) from a raw 1024x512 BGRA frame, per module with CUDA events. Writes <out>/<policy>.json, a row of profile_policies.csv,
Chrome traces, the frame list and one MLflow run. Example (on the box; --dry-run: CPU, tiny random batches, no data needed):
  python scripts/analysis/profile_policies.py --policy r1 --checkpoint $EXP/live_20261119_r1/checkpoints/r1d_r1_s0/model_epoch_030.pth \
      --data-dir $WORKSPACE/dataset_all/wor_trajectories --cache $WORKSPACE/pooled_ov0/<prefix>,$WORKSPACE/pooled_cov_ov0/<prefix> \
      --detector $WORKSPACE/checkpoints/det_visns/detector_best.pt --controller brake --batch 8 32
"""
import argparse, csv, gc, hashlib, importlib.util, json, os, platform, re, shutil, statistics, subprocess, sys, tempfile, threading, time  # noqa: E401
from pathlib import Path
import numpy as np
import torch
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)) if str(REPO) not in sys.path else None
from src.config import paths  # noqa: E402
KINDS = {"e15": "wor", "r0ftf": "wor", "r1": "wor", "r1w4": "wor", "a86_tfpp_cam": "tfpp"}
RAW_HW = (512, 1024)   # the PDM-Lite / Bench2Drive front camera (src/config/camera.py)
ENCODERS = ("CarlaPretrainedEncoder", "AIMBackbone")   # FlopCounterMode names of the frozen encoders
SUMMARY = ("params_trainable params_frozen train_gflops_per_sample train_samples_per_s train_step_ms_p50 train_step_ms_p95 train_data_wait_frac "
           "vram_peak_allocated_gb host_rss_peak_gb disk_read_mb_s_process projected_gpu_hours train_total_pflop infer_total_ms_p50 infer_total_ms_p95 "
           "infer_encoder_ms_p50 infer_detector_ms_p50 infer_head_ms_p50 infer_controller_and_io_ms_p50 infer_fps_p50 infer_gflops_per_frame "
           "encoder_gflops_per_frame mlflow_run_id").split()
def die(msg):
    raise SystemExit(f"[profile_policies] {msg}")
def need(p, what):
    if not p or not os.path.exists(p): die(f"{what} not found: {p!r}")
    return p
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, kw in (("--policy", dict(required=True, help=f"row label ({', '.join(KINDS)}; another name needs --kind)")),
                     ("--kind", dict(choices=["wor", "tfpp"], help="default: from --policy, else from the run_config")),
                     ("--checkpoint", dict(required=True, help="model_epoch_*.pth (WoR) or snapshot_*.pth (A86)")),
                     ("--run-config", dict(help="default: run_config.json beside the checkpoint")),
                     ("--frozen-backbone", dict(help="E15's frozen_backbone.pth (default: beside the checkpoint)")),
                     ("--detector", dict(help="R1 / R1+w4: the train_object_detector.py checkpoint (det_visns/detector_best.pt)")),
                     ("--object-pca-basis", dict(help="R1: default = the basis stored in the policy checkpoint (obj_pca_*)")),
                     ("--data-dir", dict(help="the run's PDM-Lite data dir")), ("--cache", dict(help="comma-separated pooled-cache prefixes")),
                     ("--frame-list", dict(help="file of frame keys (rgb path relative to --data-dir) to train on; default: a seeded random "
                                                "subset of the run's training split, written to <out>/<policy>_frames.txt")),
                     ("--batch-size", dict(type=int, help="default: the run's")), ("--num-workers", dict(type=int, help="default: the run's")),
                     ("--batch", dict(type=int, nargs="*", default=[], help="extra forward-throughput batch sizes, e.g. 8 32")),
                     ("--frame", dict(help="an RGB image to drive on (resized to 1024x512); default random pixels (same cost)")),
                     ("--controller", dict(help="WOR_CONTROLLER for the WoR agent (pid / brake / ...); default unset = pid")),
                     ("--device", dict(default="cuda" if torch.cuda.is_available() else "cpu")),
                     ("--out", dict(help="default: <exp_root>/profile_<date>")), ("--experiment-name", dict(default="A88_profile_policies"))):
        p.add_argument(name, **kw)
    for k, v in (("subset-seed", 0), ("warmup", 20), ("steps", 200), ("repeats", 3), ("profile-steps", 10), ("infer-warmup", 20), ("infer-iters", 300)):
        p.add_argument(f"--{k}", type=int, default=v)
    for k in ("skip-train", "skip-infer", "no-sha256", "no-mlflow", "dry-run"):   # dry-run: CPU, tiny random batches, a few steps
        p.add_argument(f"--{k}", action="store_true")
    a = p.parse_args()
    if a.dry_run:
        a.device, a.batch_size, a.experiment_name, a.batch = "cpu", a.batch_size or 4, a.experiment_name + "_dryrun", a.batch[:1] or [2]
        a.num_workers, a.warmup, a.steps, a.repeats, a.profile_steps, a.infer_warmup, a.infer_iters = 0, 2, 3, 2, 2, 1, 3
    return a

# ---------------------------------------------------------------------------------------------------------------- measurement helpers
def sync(dev):
    if dev.type == "cuda": torch.cuda.synchronize(dev)
def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")
def count_flops(fn):
    """Total FLOPs of fn() and the per-module split (two levels deep, top 12), from FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        fn()
    per = {k: int(sum(v.values())) for k, v in fc.get_flop_counts().items() if k.count(".") <= 1 and k != "Global"}
    return int(fc.get_total_flops()), dict(sorted(per.items(), key=lambda kv: -kv[1])[:12])
def _safe(fn):
    try:
        return fn()
    except Exception:   # a loader worker exiting between listing and reading, an nvidia-smi hiccup, no /proc
        return None
def _procs():
    """psutil and this process plus its loader workers (None, [] without psutil)."""
    ps = importlib.import_module("psutil") if importlib.util.find_spec("psutil") else None
    return ps, ([ps.Process()] + ps.Process().children(recursive=True) if ps else [])
def io_read_bytes():
    """(read bytes of this process tree, system disk read bytes); None where psutil cannot tell."""
    ps, procs = _procs()
    return (None, None) if ps is None else (sum(getattr(_safe(p.io_counters), "read_bytes", 0) for p in procs),
                                            getattr(ps.disk_io_counters(), "read_bytes", None))
def page_cache_gb():
    return _safe(lambda: next(int(ln.split()[1]) / 1e6 for ln in open("/proc/meminfo") if ln.startswith("Cached:")))
class Sampler(threading.Thread):
    """Host RSS of the process tree, CPU utilisation (every 0.5 s) and, on CUDA, nvidia-smi utilisation / memory (every 1 s)."""
    def __init__(self, dev):
        super().__init__(daemon=True)
        self.gpu, self.smi = ((dev.index or 0) if dev.type == "cuda" else None), shutil.which("nvidia-smi")
        self.halt, self.rss, self.cpu, self.util, self.mem = threading.Event(), 0, [], [], 0.0
    def _smi(self):
        q = subprocess.run([self.smi, f"--id={self.gpu}", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5).stdout
        u, m = (float(x) for x in q.strip().split(","))
        self.util.append(u)
        self.mem = max(self.mem, m / 1024)
    def run(self):
        ps, n = _procs()[0], 0
        _safe(lambda: ps.cpu_percent(None))
        while not self.halt.wait(0.5):
            n += 1
            if ps:
                self.rss = max(self.rss, _safe(lambda: sum(p.memory_info().rss for p in _procs()[1])) or 0)
                self.cpu.append(ps.cpu_percent(None))
            if self.gpu is not None and self.smi and n % 2 == 0: _safe(self._smi)
    def stop(self):
        self.halt.set()
        self.join(timeout=5)
        out = {"host_rss_peak_gb": self.rss / 1e9 or None, "cpu_util_mean_pct": float(np.mean(self.cpu)) if self.cpu else None}
        if self.gpu is not None:
            out.update(gpu_util_mean_pct=float(np.mean(self.util)) if self.util else None, gpu_util_p95_pct=pct(self.util, 95), nvsmi_mem_peak_gb=self.mem or None)
        return out
class TimedLoader:
    """Wraps a DataLoader: the wait inside next() and the compute between batches (the training loops synchronise before they ask for
    the next batch). Cycles when the subset runs out; calls prof.step() at every step boundary when profiling."""
    def __init__(self, loader, prof=None):
        self.loader, self.prof, self.wait, self.compute, self.cycles = loader, prof, [], [], 0
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        it, last = iter(self.loader), None
        while True:
            t0 = time.perf_counter()
            if last is not None:
                self.compute.append(t0 - last)
                _ = self.prof.step() if self.prof is not None else None
            b = next(it, None)
            if b is None:
                self.cycles, it = self.cycles + 1, iter(self.loader)
                b = next(it)
            last = time.perf_counter()
            self.wait.append(last - t0)
            yield b
def block_stats(tl, a, bs):
    blocks = []
    for lo in range(a.warmup, a.warmup + a.repeats * a.steps, a.steps):
        w, c = tl.wait[lo:lo + a.steps], tl.compute[lo:lo + a.steps]
        st = [x + y for x, y in zip(w, c)]
        if st: blocks.append({"step_ms_p50": pct(st, 50) * 1e3, "step_ms_p95": pct(st, 95) * 1e3, "samples_per_s": bs * len(st) / sum(st),
                              "data_wait_ms_p50": pct(w, 50) * 1e3, "compute_ms_p50": pct(c, 50) * 1e3, "data_wait_frac": sum(w) / sum(st)})
    if not blocks: die("no measured steps (check --warmup / --steps)")
    return {k: statistics.median(b[k] for b in blocks) for k in blocks[0]}, blocks
def trace_summary(prof, n, wall_s):
    ka = prof.key_averages()
    dl = sum(e.cpu_time_total for e in ka if e.key.startswith("enumerate(DataLoader)"))
    gpu = sum(getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0 for e in ka)
    return {"trace_wall_ms_per_step": wall_s * 1e3 / n, "trace_dataloader_ms_per_step": dl / 1e3 / n, "trace_gpu_kernel_ms_per_step": gpu / 1e3 / n,
            "trace_gpu_busy_frac": gpu / 1e6 / wall_s if wall_s > 0 else None, "trace_flops_per_step": sum(int(getattr(e, "flops", 0) or 0) for e in ka) / n}
def profiler(dev, **kw):
    from torch.profiler import ProfilerActivity, profile
    return profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else []), record_shapes=True,
                   profile_memory=True, with_flops=True, **kw)
def profile_training(run_steps, model, bs, dev, a, out_dir, tag):
    """run_steps(n, prof) runs n training steps through the policy's own loop and returns its TimedLoader."""
    tot, tr = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops, per = count_flops(lambda: run_steps(1, None))
    res = {"params_trainable": tr, "params_frozen": tot - tr, "params_total": tot, "batch_size": bs, "train_flops_per_sample": flops / bs,
           "train_gflops_per_sample": flops / bs / 1e9, "train_flops_top_modules": {k: v / bs for k, v in per.items()}}
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    smp, io0, pc0, t0 = Sampler(dev), io_read_bytes(), page_cache_gb(), time.perf_counter()
    smp.start()
    tl = run_steps(a.warmup + a.steps * a.repeats, None)
    wall, io1 = time.perf_counter() - t0, io_read_bytes()
    med, blocks = block_stats(tl, a, bs)
    res.update(smp.stop(), **{f"train_{k}": v for k, v in med.items()}, train_blocks=blocks, train_subset_cycles=tl.cycles, timed_wall_s=wall,
               page_cache_gb_before=pc0, page_cache_gb_after=page_cache_gb())
    for i, k in ((0, "disk_read_mb_s_process"), (1, "disk_read_mb_s_system")):
        if io0[i] is not None and io1[i] is not None: res[k] = (io1[i] - io0[i]) / 1e6 / wall
    if dev.type == "cuda":
        res.update(vram_peak_allocated_gb=torch.cuda.max_memory_allocated(dev) / 1e9, vram_peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 1e9)
    from torch.profiler import schedule
    n, trace, got = max(1, a.profile_steps), os.path.join(out_dir, f"{tag}_train_trace.json"), {}
    ready = lambda p: (p.export_chrome_trace(trace), got.update(prof=p))  # noqa: E731
    with profiler(dev, schedule=schedule(wait=1, warmup=2, active=n, repeat=1), on_trace_ready=ready) as prof:
        tlp = run_steps(3 + n + 1, prof)
    if "prof" in got: res.update(trace_summary(got["prof"], n, sum(tlp.wait[3:3 + n]) + sum(tlp.compute[3:3 + n])), train_trace=trace)
    return res
def pick_subset(a, keys, bs, out_path):
    """Positions into the training split (--frame-list keys, else a seeded random subset long enough for every phase), saved to out_path."""
    want = {ln.strip() for ln in open(need(a.frame_list, "--frame-list")) if ln.strip()} if a.frame_list else None
    n = (a.warmup + a.steps * a.repeats + a.profile_steps + 5) * bs
    idx = [j for j, k in enumerate(keys) if k in want] if want else np.random.default_rng(a.subset_seed).permutation(len(keys))[:n].tolist()
    if not idx: die("the frame subset is empty (does --frame-list use keys relative to --data-dir?)")
    idx = idx[:max(bs, len(idx) // bs * bs)]
    _ = out_path and open(out_path, "w").writelines(keys[j] + "\n" for j in idx)
    return idx

# ------------------------------------------------------------------------------------------------------------------- WoR / Qwen policies
def synthetic_wor(n, ch, grid, m_obj, rp, n_wp):
    """--dry-run: samples in the pooled-cache format WorldOnRailsDataset serves (R1: flat [grid | detector tokens])."""
    out, z = [], torch.zeros
    for i in range(n):
        feats, tok = torch.randn(ch, *grid), torch.rand(m_obj, 16)
        tok[:, 15] = (tok[:, 15] > 0.5).float()   # build_box_targets.VALID_SLOT
        out.append({"vision_features": torch.cat([feats.reshape(-1), tok.reshape(-1)]) if m_obj else feats, "speed": torch.rand(1) * 10, "command": torch.tensor(3),
                    "route": torch.randn(rp, 2), "target_q": z(9), "target_waypoints": torch.randn(n_wp, 2), "target_speed": torch.rand(()) * 10,
                    "target_path": z(10, 2), "path_valid": z(()), "target_qsa": z(40), "qsa_valid": z(()), "swerve": z(()), "brake_onset": z(()),
                    "scenario": torch.tensor(-1)})
    return out
def wor_obj_files(a, ck, sd, tmp):
    """R1: detector checkpoint and PCA basis. The basis defaults to the one stored in the policy checkpoint (what it trained with); a dry
    run without --detector rebuilds the detector file from the checkpoint's obj_det.* tensors."""
    basis, det = a.object_pca_basis or os.path.join(tmp, "obj_pca_basis.pt"), a.detector
    _ = a.object_pca_basis or torch.save({"mean": sd["obj_pca_mean"], "basis": sd["obj_pca_basis"]}, basis)
    if not det and a.dry_run:
        det, dc = os.path.join(tmp, "detector_from_ckpt.pt"), dict((ck.get("config") or {}).get("object_det_cfg") or {})
        torch.save({"config": {**dc, "pca": dc.get("in_ch", 256)}, "state_dict": {k[8:]: v for k, v in sd.items() if k.startswith("obj_det.")}}, det)
    return need(det, "--detector (R1 detector checkpoint)"), need(basis, "--object-pca-basis")
def wor_training(a, cfg, ck, dev, tmp, out_dir, tag):
    import scripts.training.train_wor as tw
    import src.training.wor_trainer as wt
    from src.training.pooled_feature_cache import rel_key
    from src.training.wor_dataloaders import _wrap_loader
    sd = {re.sub(r"^(module\.|model\.|_orig_mod\.)+", "", k): v for k, v in (ck.get("model") or ck.get("state_dict") or {}).items()}
    m_obj, grid, bs = int(cfg.get("object_tokens") or 0), tuple(int(x) for x in (ck.get("config") or {}).get("vision_grid", [4, 4])), int(a.batch_size or cfg["batch_size"])
    over = {"save_dir": os.path.join(tmp, "trainer"), "use_mlflow": 0, "kill_stale": 0, "auto_batch_size": 0, "device": dev.type, "resume_from": None,
            "train_eval_frames": 0, "batch_size": bs, "weights_path": need(a.frozen_backbone, "--frozen-backbone"),
            "num_workers": a.num_workers if a.num_workers is not None else cfg.get("num_workers")}
    if m_obj: over["object_detector"], over["object_pca_basis"] = wor_obj_files(a, ck, sd, tmp)
    syn = None
    if a.dry_run:   # metadata-only fake caches, so main()'s checks see the run's layout; the batches are synthetic
        prefixes = [os.path.join(tmp, f"fake_cache{i}") for i in range(len((cfg.get("pooled_cache") or "x").split(",")))]
        for pre in prefixes:
            json.dump({"grid": list(grid)}, open(pre + ".json", "w"))
            _ = m_obj and json.dump({"max_range": cfg.get("object_max_range", 30.0), "threshold": cfg.get("object_threshold", 0.5)},
                                    open(pre + f".boxtok{m_obj}{cfg.get('object_variant', '_det')}.f16.json", "w"))
        syn = synthetic_wor(bs * 8, int(sd["vision_proj.weight"].shape[1]), grid, m_obj, int(cfg.get("route_points", 4)), 10)
        over.update(data_dir=os.path.join(tmp, "no_data"), pooled_cache=",".join(prefixes) if cfg.get("pooled_cache") else None)
        wt.create_wor_train_val_dataloaders = lambda **kw: (_wrap_loader(syn, kw["batch_size"], 0, True, kw.get("seed")), None)
    else:
        over["data_dir"] = need(a.data_dir, "--data-dir")
        if cfg.get("pooled_cache"):
            over["pooled_cache"] = ",".join(need(pre + ".json", "pooled cache metadata")[:-5] for pre in need(a.cache, "--cache").split(","))
    sys.argv = ["train_wor.py"]
    dests, argv, got = set(vars(tw.parse_args())), ["train_wor.py"], {}
    for k, v in {**cfg, **over}.items():
        if k in dests and v is not None and v != "": argv += [f"--{k}", str(v)]
    class Capture(tw.WorldOnRailsTrainer):   # everything up to the training loop is the run's own; the loop is measured instead
        def train(self, num_epochs=50, save_freq=5, resume_from=None, stop_epoch=None):
            got.update(trainer=self, num_epochs=num_epochs)
    tw.WorldOnRailsTrainer, sys.argv = Capture, argv
    tw.main()
    tr = got["trainer"]
    _, unexpected = tr.model.load_state_dict(sd, strict=False)   # the trained heads, so the experts route as they did
    print(f"--> heads from {a.checkpoint}: {len(sd) - len(unexpected)} tensors loaded, {len(unexpected)} unexpected", flush=True)
    if syn is not None:   # match the waypoint count the head predicts
        b = syn[0]
        with torch.no_grad():
            n_wp = tr.model.eval()(None, b["speed"][None], b["command"][None], b["route"][None], vision_features=b["vision_features"][None])["selected_waypoints"].shape[1]
        [s.update(target_waypoints=torch.randn(n_wp, 2)) for s in syn]
        tr.model.train()
    ds = tr.train_loader.dataset
    base, pos = getattr(ds, "dataset", ds), list(getattr(ds, "indices", range(len(ds))))
    keys = [str(j) for j in pos] if syn is not None else [rel_key(base.samples[i]["rgb_path"], over["data_dir"]) for i in pos]
    idx = pick_subset(a, keys, bs, None if syn is not None else os.path.join(out_dir, f"{tag}_frames.txt"))
    info = {"samples_per_epoch": len(ds), "steps_per_epoch": len(tr.train_loader), "frames_indexed": len(base), "subset_frames": len(idx),
            "precision": "bf16 autocast" if tr.use_amp else "fp32", "optimizer": type(tr.optimizer).__name__,
            "lr_groups": [g["lr"] for g in tr.optimizer.param_groups], "num_workers": int(over["num_workers"] or 0)}
    tr.scheduler = tr._build_scheduler(got["num_epochs"])   # the run's schedule (built on the full loader)
    loader = _wrap_loader(torch.utils.data.Subset(ds, idx), bs, info["num_workers"], False, int(cfg.get("seed", 0)))
    def run_steps(n, prof):
        tr.train_loader, tr.max_batches = TimedLoader(loader, prof), n
        tr.train_epoch(1)
        return tr.train_loader
    return {**profile_training(run_steps, tr.model, bs, dev, a, out_dir, tag), **info}
def slim_checkpoint(a, ck, cfg_path, tmp):
    """The agent loads <dir>/model.pth with run_config.json and frozen_backbone.pth beside it. Rewritten with weights, config and flags only,
    so the loader's weights_only torch.load reads the stamped config instead of silently falling back to defaults."""
    d = os.path.join(tmp, "agent")
    os.makedirs(d, exist_ok=True)
    keep = {**{k: ck[k] for k in ("model", "config", "partial", "epoch") if k in ck}, **({"frozen_ref": "frozen_backbone.pth"} if ck.get("partial") else {})}
    _ = ck.get("partial") and shutil.copyfile(need(a.frozen_backbone, "--frozen-backbone"), os.path.join(d, "frozen_backbone.pth"))
    torch.save(keep, os.path.join(d, "model.pth"))
    shutil.copyfile(cfg_path, os.path.join(d, "run_config.json"))
    return os.path.join(d, "model.pth")
class Events:
    """Per-module timing: CUDA events around wrapped calls (perf_counter on CPU), summed per frame by take()."""
    def __init__(self, dev):
        self.cuda, self.pairs = dev.type == "cuda", {}
    def mark(self):
        e = torch.cuda.Event(enable_timing=True) if self.cuda else None
        return e.record() or e if self.cuda else time.perf_counter()
    def wrap(self, name, fn):
        def w(*args, **kw):
            s, r = self.mark(), fn(*args, **kw)
            self.pairs.setdefault(name, []).append((s, self.mark()))
            return r
        return w
    def take(self):
        out, self.pairs = {n: sum((s.elapsed_time(e) if self.cuda else (e - s) * 1e3) for s, e in ps) for n, ps in self.pairs.items()}, {}
        return out
def raw_frame(a):
    from PIL import Image
    rgb = (np.asarray(Image.open(need(a.frame, "--frame")).convert("RGB").resize(RAW_HW[::-1])) if a.frame
           else np.random.default_rng(0).integers(0, 256, (*RAW_HW, 3), dtype=np.uint8))
    return np.concatenate([rgb[..., ::-1], np.full((*RAW_HW, 1), 255, np.uint8)], axis=-1)   # CARLA delivers BGRA
def measure_inference(frame_fn, ev, dev, a, out_dir, tag, batched_fn):
    """frame_fn() drives one frame and returns wall seconds {'total', 'preprocess'}; ev holds that frame's module events."""
    for _ in range(a.infer_warmup):
        frame_fn(), sync(dev), ev.take()
    flops, per = count_flops(frame_fn)
    sync(dev), ev.take()
    if dev.type == "cuda": torch.cuda.reset_peak_memory_stats(dev)
    rows = []
    for _ in range(a.infer_iters):
        sync(dev)
        w, _ = frame_fn(), sync(dev)
        m = ev.take()
        rows.append({"total": w["total"] * 1e3, "preprocess": w["preprocess"] * 1e3, **m, "head": m.get("forward", 0.0) - m.get("encoder", 0.0) - m.get("detector", 0.0),
                     "controller_and_io": (w["total"] - w["preprocess"]) * 1e3 - m.get("forward", 0.0)})
    res = {"infer_gflops_per_frame": flops / 1e9, "infer_flops_top_modules": per, "infer_iters": len(rows)}
    res.update({f"infer_{k}_ms_p{q}": pct([r[k] for r in rows if k in r], q) for k in rows[0] for q in (50, 95, 99)})
    res["infer_fps_p50"], res["infer_trace"] = 1e3 / res["infer_total_ms_p50"], os.path.join(out_dir, f"{tag}_infer_trace.json")
    if dev.type == "cuda":
        res.update(infer_vram_peak_allocated_gb=torch.cuda.max_memory_allocated(dev) / 1e9, infer_vram_peak_reserved_gb=torch.cuda.max_memory_reserved(dev) / 1e9)
    with profiler(dev) as prof:
        [(frame_fn(), sync(dev)) for _ in range(min(10, a.infer_iters))]
    ev.take(), prof.export_chrome_trace(res["infer_trace"])
    with torch.no_grad():
        for b in a.batch:
            t = []
            for i in range(3 + max(3, a.infer_iters // 10)):   # 3 warm-up calls
                sync(dev)
                t0 = time.perf_counter()
                batched_fn(b), sync(dev)
                _ = i >= 3 and t.append(time.perf_counter() - t0)
            res[f"infer_batch{b}_ms_p50"], res[f"infer_batch{b}_frames_per_s"] = pct(t, 50) * 1e3, b / pct(t, 50)
    ev.take()
    return res
def wor_inference(a, cfg, ck, cfg_path, dev, tmp, out_dir, tag):
    if a.controller: os.environ["WOR_CONTROLLER"] = a.controller
    import src.agents.wor_agent as wa
    agent = wa.WorldOnRailsAgent(checkpoint_path=slim_checkpoint(a, ck, cfg_path, tmp), backbone_name=cfg.get("backbone", "regnety_032"),
                                 policy_arch=cfg.get("policy_arch", "cnn"), device=dev.type)
    net, ev, pre_t, orig_pre = agent.net, Events(dev), [], wa.preprocess_rgb
    net.encoder.forward = ev.wrap("encoder", net.encoder.forward)
    if int(getattr(net, "object_tokens", 0) or 0): net.object_tokens_from_map = ev.wrap("detector", net.object_tokens_from_map)
    net.forward = ev.wrap("forward", net.forward)
    def timed_pre(*args, **kw):
        t0, r = time.perf_counter(), orig_pre(*args, **kw)
        pre_t.append(time.perf_counter() - t0)
        return r
    wa.preprocess_rgb, rp, (h, w) = timed_pre, int(agent.route_points), agent.img_size
    route = np.stack([np.linspace(4.0, 4.0 * rp, rp), np.zeros(rp)], 1).astype(np.float32)
    data = {"rgb_front": (0, raw_frame(a)), "speed": (0, {"speed": 6.0}), "command": 4, "route": route}
    def frame():
        pre_t.clear()
        t0, _ = time.perf_counter(), agent.run_step(data, 0.0)
        return {"total": time.perf_counter() - t0, "preprocess": sum(pre_t)}
    def batched(b):
        net(torch.rand(b, h, w, 3, device=dev) * 255, torch.full((b, 1), 6.0, device=dev), torch.full((b,), 3, device=dev, dtype=torch.long),
            torch.as_tensor(route, device=dev).expand(b, -1, -1))
    res = measure_inference(frame, ev, dev, a, out_dir, tag, batched)
    wa.preprocess_rgb = orig_pre
    return {**res, "infer_img_hw": f"{h}x{w}", "infer_controller": os.environ.get("WOR_CONTROLLER", "pid"), "infer_precision": "fp32 (agent act)"}

# ------------------------------------------------------------------------------------------------------------- A86 TF++ camera-only
def tfpp_setup(cfg, ck, dev):
    from src.baselines import tfpp_cam as tc
    h, w = (int(x) for x in cfg.get("img_size", "288x768").split("x"))
    config = tc.make_config((h, w))
    config.epochs, config.lr, config.batch_size = int(cfg.get("epochs", 31)), float(cfg.get("lr", 3e-4)), int(cfg.get("batch_size", 128))
    net = tc.build_model(config, None)
    net.load_state_dict(ck["model"], strict=True)   # the snapshot is the full LidarCenterNet, frozen encoder included
    return tc, config, net.to(dev), (h, w)
def tfpp_training(a, cfg, ck, dev, tmp, out_dir, tag):
    import scripts.training.train_tfpp_cam as tt
    tc, config, net, (h, w) = tfpp_setup(cfg, ck, dev)
    tc.use_cached_maps(net)
    net.train()
    bs = int(a.batch_size or cfg.get("batch_size", 128))
    if a.dry_run:   # synthetic samples in TfppCamFrames' format
        ns, ch, hw = len(config.target_speeds), int(net.backbone.num_image_features), (config.img_vert_anchors, config.img_horz_anchors)
        ds = [{"map": torch.randn(ch, *hw), "augmented": False, "checkpoint": torch.randn(config.predict_checkpoint_len, 2), "target_point": torch.randn(2),
               "target_speed_twohot": torch.eye(ns)[i % ns], "target_speed_index": torch.tensor(i % ns), "speed": torch.tensor(5.0),
               "command": torch.eye(6)[3]} for i in range(bs * 8)]
        info = {"frames_indexed": len(ds), "subset_frames": len(ds)}
    else:
        from src.training.pooled_feature_cache import PooledFeatureCache, prefixes_from_env, rel_key
        ns = argparse.Namespace(data_dir=need(a.data_dir, "--data-dir"), img_size=f"{h}x{w}", val_split=float(cfg.get("val_split", 0.15)),
                                split_seed=int(cfg.get("split_seed", 0)))
        [need(pre + ".json", "pooled cache metadata") for pre in need(a.cache, "--cache").split(",")]
        train_paths, _, ptag = tt.frame_lists(ns)   # train_tfpp_cam.py's frame list and route split
        cache = tt._CacheView(PooledFeatureCache(prefixes_from_env(a.cache), ptag, ns.data_dir))
        idx = pick_subset(a, [rel_key(p, ns.data_dir) for p in train_paths], bs, os.path.join(out_dir, f"{tag}_frames.txt"))
        info = {"samples_per_epoch": len(train_paths) // bs * bs, "steps_per_epoch": len(train_paths) // bs, "frames_indexed": len(cache.c.index),
                "subset_frames": len(idx)}
        ds = tt.TfppCamFrames([train_paths[j] for j in idx], ns.data_dir, cache, tc.labels_for(config), True, config.augment_percentage)
    nw = 0 if a.dry_run else int(a.num_workers if a.num_workers is not None else cfg.get("num_workers", 6))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=nw, pin_memory=dev.type == "cuda", persistent_workers=nw > 0, drop_last=True,
                                         shuffle=False, **({"prefetch_factor": 4} if nw else {}))   # train_tfpp_cam.py's loader settings
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=config.lr, amsgrad=True)   # and its optimizer / schedule
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config.cosine_t0, T_mult=config.cosine_t_mult)
    iters = max(1, info.get("steps_per_epoch", len(loader)))
    def run_steps(n, prof):   # train_tfpp_cam.py's loop body
        tl = TimedLoader(loader, prof)
        for i, b in enumerate(tl):
            if i >= n: break
            int(b["augmented"].sum())
            b = tt.to_dev(b, dev)
            loss = tc.planning_loss(net, tt.forward(net, b), b)
            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            opt.step(), sched.step(i / iters)
            [float(loss[k].detach()) for k in ("total", "loss_checkpoint", "loss_target_speed")]
            sync(dev)
        return tl
    return {**profile_training(run_steps, net, bs, dev, a, out_dir, tag), **info, "num_workers": nw, "precision": "fp32", "optimizer": "AdamW(amsgrad)"}
def tfpp_inference(a, cfg, ck, dev, out_dir, tag):
    from src.config.camera import preprocess_rgb
    _, config, net, (h, w) = tfpp_setup(cfg, ck, dev)
    ev, raw, speeds = Events(dev), raw_frame(a), list(config.target_speeds)
    net.eval().backbone.forward, fwd = ev.wrap("encoder", net.backbone.forward), ev.wrap("forward", net.forward)
    tp, vel, cmd = torch.tensor([[8.0, 0.0]], device=dev), torch.tensor([[6.0]], device=dev), torch.eye(6, device=dev)[3:4]
    def frame():
        t0 = time.perf_counter()
        img = preprocess_rgb(np.ascontiguousarray(raw[..., [2, 1, 0]]), img_size=(h, w), crop_bottom_frac=0.0)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = fwd(rgb=torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(dev), lidar_bev=None, target_point=tp, ego_vel=vel, command=cmd)
            net.control_pid_direct(out[2][0].detach().cpu().numpy(), speeds[int(torch.argmax(torch.softmax(out[1][0], 0)))], vel[0])
        return {"total": time.perf_counter() - t0, "preprocess": t1 - t0}
    def batched(b):
        net(rgb=torch.rand(b, 3, h, w, device=dev) * 255, lidar_bev=None, target_point=tp.expand(b, -1), ego_vel=vel.expand(b, -1), command=cmd.expand(b, -1))
    return {**measure_inference(frame, ev, dev, a, out_dir, tag, batched), "infer_img_hw": f"{h}x{w}", "infer_precision": "fp32",
            "infer_controller": "TF++ control_pid_direct (argmax target speed)",
            "infer_note": "excludes the TF++ agent's JPEG round trip and stop-sign / stuck logic in tick()"}

# ------------------------------------------------------------------------------------------------------------------------- main
def telemetry_crosscheck(ck_dir):
    """What the original run logged: epochs, mean epoch seconds, mean samples/s (wor_training_telemetry.csv / tfpp_cam_telemetry.csv)."""
    for name, col in (("wor_training_telemetry.csv", "epoch_time_sec"), ("tfpp_cam_telemetry.csv", "epoch_sec")):
        p = os.path.join(ck_dir, name)
        rows = list(csv.DictReader(open(p))) if os.path.exists(p) else None
        if rows is None: continue
        t, sps = [float(r[col]) for r in rows if r.get(col)], [float(r["samples_per_sec"]) for r in rows if r.get("samples_per_sec")]
        return {"logged_epochs": len(rows), "logged_epoch_s_mean": float(np.mean(t)) if t else None,
                "logged_samples_per_s_mean": float(np.mean(sps)) if sps else None, "logged_from": p}
    return {}
def recipe_totals(res, cfg, kind):
    """Epochs this run trained (a resumed run counts from its resume epoch) and the projected cost of that recipe on this GPU."""
    m = re.search(r"model_epoch_(\d+)", str(cfg.get("resume_from") or ""))
    n = (int(cfg.get("stop_epoch") or cfg.get("epochs") or 0) - (int(m.group(1)) if m else 0) if kind == "wor"
         else int(res.get("logged_epochs") or cfg.get("epochs") or 0))
    res.update(recipe_epochs=n, recipe_resumed_from=cfg.get("resume_from"))
    spe, ms = res.get("steps_per_epoch"), res.get("train_step_ms_p50")
    if spe and ms: res.update(projected_epoch_s=spe * ms / 1e3, projected_gpu_hours=spe * ms / 1e3 * n / 3600)
    if res.get("samples_per_epoch") and res.get("train_flops_per_sample"):
        res["train_total_pflop"] = res["train_flops_per_sample"] * res["samples_per_epoch"] * n / 1e15
    enc = [v for k, v in (res.get("infer_flops_top_modules") or {}).items() if k in ENCODERS]
    if enc and res.get("frames_indexed"): res.update(encoder_gflops_per_frame=enc[0] / 1e9, cache_build_pflop=enc[0] * res["frames_indexed"] / 1e15)
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, f"{pre}{k}."))
        elif not isinstance(v, list): out[pre + k] = v
    return out
def write_outputs(res, out_dir, tag):
    jp, cp = os.path.join(out_dir, f"{tag}.json"), os.path.join(out_dir, "profile_policies.csv")
    json.dump(res, open(jp, "w"), indent=2, default=str)
    rows = [r for r in (list(csv.DictReader(open(cp))) if os.path.exists(cp) else []) if r.get("tag") != tag] + [flat(res)]
    with open(cp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        wr.writeheader(), wr.writerows(rows)
    return jp, cp
def log_mlflow(a, res, files):
    import mlflow
    mlflow.set_tracking_uri(paths.mlflow_tracking_uri())   # also sets MLFLOW_ALLOW_FILE_STORE=true
    mlflow.set_experiment(a.experiment_name)
    num, fl = (lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)), flat(res)
    with mlflow.start_run(run_name=res["tag"]) as run:
        mlflow.log_params({k[:250]: str(v)[:500] for k, v in fl.items() if not num(v)})
        mlflow.log_metrics({re.sub(r"[^\w./ -]", "_", k)[:250]: float(v) for k, v in fl.items() if num(v) and np.isfinite(v)})
        [mlflow.log_artifact(f) for f in files if f and os.path.exists(f)]
        return run.info.run_id
def main():
    a = parse_args()
    ck_dir = os.path.dirname(os.path.abspath(need(a.checkpoint, "--checkpoint")))
    cfg_path = need(a.run_config or os.path.join(ck_dir, "run_config.json"), "run_config.json")
    cfg = json.load(open(cfg_path))
    kind = a.kind or KINDS.get(a.policy.lower()) or ("wor" if "policy_arch" in cfg else "tfpp" if "frozen_backbone" in cfg else None)
    if kind is None: die(f"cannot tell the policy kind of {a.policy!r}: pass --kind wor|tfpp")
    if kind == "wor" and not a.frozen_backbone and os.path.exists(os.path.join(ck_dir, "frozen_backbone.pth")):
        a.frozen_backbone = os.path.join(ck_dir, "frozen_backbone.pth")
    if not a.no_mlflow and importlib.util.find_spec("mlflow") is None: die("mlflow is not installed (pip install mlflow-skinny), or pass --no-mlflow")
    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available(): die("--device cuda but CUDA is not available (use --dry-run for a CPU check)")
    tag, out_dir = a.policy + ("_dryrun" if a.dry_run else ""), a.out or str(paths.exp_root() / f"profile_{time.strftime('%Y-%m-%d')}")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    print(f"--> A88 profile {a.policy} ({kind}) on {dev}{' [DRY RUN]' if a.dry_run else ''}; output {out_dir}", flush=True)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    res = {"tag": tag, "policy": a.policy, "kind": kind, "dry_run": a.dry_run, "checkpoint": os.path.abspath(a.checkpoint),
           "checkpoint_mb": os.path.getsize(a.checkpoint) / 1e6, "checkpoint_epoch": ck.get("epoch"), "run_config": cfg_path,
           "run_label": cfg.get("run_label") or os.path.basename(ck_dir), "data_dir": a.data_dir, "cache": a.cache, "torch": torch.__version__,
           "cuda": torch.version.cuda, "device": str(dev), "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
           "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, "controller_env": a.controller,
           "protocol": {k: getattr(a, k) for k in ("warmup", "steps", "repeats", "profile_steps", "infer_warmup", "infer_iters", "subset_seed")}}
    if not a.no_sha256:
        h = hashlib.sha256()
        with open(a.checkpoint, "rb") as f:
            [h.update(c) for c in iter(lambda: f.read(1 << 24), b"")]
        res["checkpoint_sha256"] = h.hexdigest()
    if a.cache: res["cache_disk_gb"] = sum(os.path.getsize(p + ".f16") for p in a.cache.split(",") if os.path.exists(p + ".f16")) / 1e9
    res.update(telemetry_crosscheck(ck_dir))
    tmp, argv0 = tempfile.mkdtemp(prefix="a88_profile_"), list(sys.argv)
    try:
        if not a.skip_train:
            res.update((wor_training if kind == "wor" else tfpp_training)(a, cfg, ck, dev, tmp, out_dir, tag))
            sys.argv = argv0
            gc.collect()   # the trainer, its optimizer state and loader workers must not sit in VRAM / RAM during inference
            if dev.type == "cuda": torch.cuda.empty_cache()
        if not a.skip_infer:
            if dev.type == "cuda": res["infer_vram_baseline_gb"] = torch.cuda.memory_allocated(dev) / 1e9   # left over before the agent (want ~0)
            res.update(wor_inference(a, cfg, ck, cfg_path, dev, tmp, out_dir, tag) if kind == "wor" else tfpp_inference(a, cfg, ck, dev, out_dir, tag))
    finally:
        sys.argv = argv0
        shutil.rmtree(tmp, ignore_errors=True)
    recipe_totals(res, cfg, kind)
    jp, cp = write_outputs(res, out_dir, tag)
    if not a.no_mlflow:
        res["mlflow_run_id"] = log_mlflow(a, res, [jp, res.get("train_trace"), res.get("infer_trace"), os.path.join(out_dir, f"{tag}_frames.txt")])
        jp, cp = write_outputs(res, out_dir, tag)
    for k in SUMMARY:
        if res.get(k) is not None: print(f"    {k:32s} {res[k]:.4g}" if isinstance(res[k], float) else f"    {k:32s} {res[k]}", flush=True)
    print(f"--> wrote {jp} and {cp}", flush=True)
if __name__ == "__main__":
    main()
