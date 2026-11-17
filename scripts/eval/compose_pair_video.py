#!/usr/bin/env python3
"""Side-by-side closed-loop videos of two policies on the same route, with the policy token's attention on the camera image
and the decision behind every control (2026-11-12, user request: best model vs the A81 arm on 10 successful + 10 failing routes).

Input: two route folders written by scripts/eval/bench2drive_agent.py with B2D_VIDEO_DIR (+ the attention probe,
src/eval/attention_probe.py): NNNNNN.jpg frames, ticks.jsonl (control, speed, target, speed posterior, waypoints, per-layer
attention of the policy token, speed-head experts) and meta.json (token layout, grid size, recording stride).

Each half of the frame shows one policy:
  header    name, the re-drive's own result (status, DS) and the scored run's result, collision banners
  camera    the frame the policy received, with the policy token's attention over the 4x4 vision cells as a heat map
            (head-averaged, mean over the 8 trunk layers; each cell's share is printed in it) and the attention split between
            vision / speed / route / command tokens
  panel     steer / throttle / brake, speed vs target, the speed-head posterior over its bins, a bird's-eye view of the route
            given and the waypoints predicted, and for speed-head experts (A81) the gate, the 2 experts used and how much they
            moved the target speed
A shared timeline at the bottom has both speed traces, both target traces, collision marks and the playback cursor.
The two drives are aligned by simulation tick (the same time since route start), not by position.

usage:
  compose_pair_video.py pair  LEFT_DIR RIGHT_DIR OUT.mp4 [--names A B] [--results "..." "..."] [--events-left F..] [--events-right F..]
  compose_pair_video.py batch --video-root R --left-sub e15 --right-sub a81 --left-results GLOB --right-results GLOB
                              --clips-root C --out-dir D [--groups groups.json] [--scored-right GLOB] [--jobs 8]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np

HALF = 960
H_HEAD, H_PANEL, H_TIME = 64, 300, 130
BG = (24, 24, 24)
TXT = (235, 235, 235)
DIM = (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX
SIDE_COLORS = ((255, 200, 80), (80, 170, 255))      # BGR: left cyan-ish, right orange
EVENT_COLORS = {"vehicle": (60, 60, 255), "walker": (200, 60, 255), "layout": (60, 160, 255), "stuck": (0, 220, 220)}


def put(img, text, x, y, scale=0.55, color=TXT, thick=1):
    cv2.putText(img, text, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def load_run(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "ticks.jsonl")) if l.strip()]
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    return {"dir": d, "rows": {r["n"]: r for r in rows}, "order": [r["n"] for r in rows], "meta": meta}


def parse_events(paths):
    """A44 clip files '<tag>_h<id>_<kind>_t<NNNN>.npz' -> [(t_seconds, kind)]."""
    out = []
    for p in paths or []:
        m = re.search(r"_(vehicle|walker|layout|stuck)_t(\d+)\.npz$", os.path.basename(p))
        if m:
            out.append((float(m.group(2)), m.group(1)))
    return sorted(out)


# ---------------------------------------------------------------- camera + attention
def attention_cells(row, layout, layer="mean"):
    a = row.get("attn")
    if not a or not layout:
        return None
    A = np.asarray(a, dtype=np.float32)
    v = A.mean(0) if layer == "mean" else A[-1]
    gh, gw = layout["grid_hw"]
    nv = layout["n_vision"]
    cells = v[1:1 + gh * gw]
    rest = v[1 + nv:]
    split = {"vision": float(v[1:1 + nv].sum()), "speed": float(rest[0]), "route": float(rest[1]), "command": float(rest[2]), "policy": float(v[0])}
    return cells.reshape(gh, gw), split


def camera_view(frame, row, layout, w, h):
    cam = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    got = attention_cells(row, layout)
    if got is None:
        put(cam, "no attention recorded", 10, 24, 0.6, (0, 200, 255), 2)
        return cam
    grid, split = got
    share = grid / max(float(grid.sum()), 1e-9)                       # each cell's share of the vision attention
    norm = share / max(float(share.max()), 1e-9)
    heat = cv2.resize(norm.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).clip(0, 1)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (0.10 + 0.50 * heat)[..., None]
    cam = (cam.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha).astype(np.uint8)
    gh, gw = grid.shape
    for i in range(1, gh):
        cv2.line(cam, (0, i * h // gh), (w, i * h // gh), (230, 230, 230), 1)
    for j in range(1, gw):
        cv2.line(cam, (j * w // gw, 0), (j * w // gw, h), (230, 230, 230), 1)
    for i in range(gh):
        for j in range(gw):
            x, y = j * w // gw + 6, i * h // gh + 20
            put(cam, f"{100 * share[i, j]:.0f}%", x + 1, y + 1, 0.5, (0, 0, 0), 2)
            put(cam, f"{100 * share[i, j]:.0f}%", x, y, 0.5, (255, 255, 255), 1)
    s = split
    label = f"policy input (route painted) | attention: vision {100 * s['vision']:.0f}%  route {100 * s['route']:.0f}%  speed {100 * s['speed']:.0f}%  cmd {100 * s['command']:.0f}%  self {100 * s['policy']:.0f}%"
    cv2.rectangle(cam, (0, h - 26), (w, h), (0, 0, 0), -1)
    put(cam, label, 8, h - 8, 0.5, (235, 235, 235))
    return cam


# ---------------------------------------------------------------- action / decision panel
def bar(img, x, y, w, h, frac, color, label):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), color, -1)
    put(img, label, x, y - 6, 0.5)


def steer_bar(img, x, y, w, h, steer):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    mid = x + w // 2
    px = int(mid + max(-1.0, min(1.0, steer)) * (w // 2))
    cv2.rectangle(img, (min(mid, px), y), (max(mid, px), y + h), (0, 200, 255), -1)
    cv2.line(img, (mid, y - 2), (mid, y + h + 2), (220, 220, 220), 1)
    put(img, f"steer {steer:+.2f}   (left - | + right)", x, y - 6, 0.5)


def posterior(img, x0, y0, w, h, row):
    probs = row.get("probs")
    dec = row.get("dec") or {}
    bins = dec.get("bins_kmh")
    put(img, "speed head: P(target speed bin)", x0, y0 + 14, 0.5)
    if not probs:
        put(img, "n/a", x0, y0 + 40, 0.5, DIM)
        return
    n = len(probs)
    top, base = y0 + 26, y0 + h - 26
    bw = max(4, (w - 8) // n)
    for i, p in enumerate(probs):
        bx = x0 + i * bw
        cv2.rectangle(img, (bx + 2, base - int((base - top) * p)), (bx + bw - 2, base), (120, 200, 120), -1)
        lab = f"{bins[i]:.0f}" if bins and i < len(bins) else str(i)
        put(img, lab, bx + 2, base + 16, 0.42, DIM)
    put(img, "km/h", x0 + n * bw + 4, base + 16, 0.42, DIM)
    if bins and len(bins) == n:   # expected target and current speed on the bin axis (piecewise-linear between bin centres)
        def xpos(kmh):
            idx = np.interp(kmh, bins, np.arange(n))
            return int(x0 + (idx + 0.5) * bw)
        tgt = row.get("target_kmh")
        if tgt is not None:
            cv2.line(img, (xpos(tgt), top - 4), (xpos(tgt), base), (60, 60, 255), 2)
        cv2.line(img, (xpos(row["speed_mps"] * 3.6), top - 4), (xpos(row["speed_mps"] * 3.6), base), (255, 255, 255), 1)


def bev(img, x0, y0, size, row, scale=5.0):
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (16, 16, 16), -1)
    cx, cy = x0 + size // 2, y0 + size - 18
    cv2.rectangle(img, (cx - 4, cy - 8), (cx + 4, cy + 8), (0, 255, 0), -1)

    def px(p):
        return int(cx + float(p[1]) * scale), int(cy - float(p[0]) * scale)

    for key, color in (("route_ego", (255, 160, 60)), ("waypoints_pred", (60, 60, 255))):
        pts = row.get(key)
        if not pts:
            continue
        pts = [px(p) for p in pts]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 3, color, -1)
    put(img, "BEV: route in (blue)", x0 + 4, y0 + 14, 0.42)
    put(img, "waypoints out (red)", x0 + 4, y0 + 30, 0.42)


def experts(img, x0, y0, w, row):
    moe = (row.get("dec") or {}).get("moe")
    if not moe:
        put(img, "speed-head experts:", x0, y0 + 14, 0.5)
        put(img, "none in this model", x0, y0 + 36, 0.5, DIM)
        return
    g = np.asarray(moe["gate"], dtype=np.float32)
    E = len(g)
    side = int(np.ceil(np.sqrt(E)))
    cell = min(34, (w - 8) // side)
    put(img, f"speed-head experts (top-{len(moe['top'])} of {E})", x0, y0 + 14, 0.5)
    gy = y0 + 24
    for e in range(E):
        r, c = divmod(e, side)
        x, y = x0 + c * cell, gy + r * cell
        v = float(g[e] / max(g.max(), 1e-9))
        col = cv2.applyColorMap(np.uint8([[int(255 * v)]]), cv2.COLORMAP_VIRIDIS)[0, 0].tolist()
        cv2.rectangle(img, (x + 1, y + 1), (x + cell - 2, y + cell - 2), col, -1)
        put(img, str(e), x + 4, y + cell // 2 + 4, 0.38, (255, 255, 255) if v < 0.6 else (0, 0, 0))
    for e, wt in zip(moe["top"], moe["top_w"]):
        r, c = divmod(e, side)
        x, y = x0 + c * cell, gy + r * cell
        cv2.rectangle(img, (x, y), (x + cell - 1, y + cell - 1), (60, 60, 255), 2)
    ty = gy + side * cell + 18
    put(img, "used: " + "  ".join(f"#{e} {wt:.2f}" for e, wt in zip(moe["top"], moe["top_w"])), x0, ty, 0.48)
    put(img, ("STANDING" if moe["standing"] else "moving") + f"   target {moe['tgt_base_kmh']:.1f} -> {moe['tgt_moe_kmh']:.1f} km/h", x0, ty + 20, 0.48,
        (0, 220, 220) if moe["standing"] else TXT)
    put(img, "(without -> with experts)", x0, ty + 38, 0.42, DIM)


def panel(row, ended):
    img = np.full((H_PANEL, HALF, 3), BG, np.uint8)
    if row is None:
        return img
    steer_bar(img, 14, 34, 270, 14, row["steer"])
    bar(img, 14, 82, 270, 14, row["throttle"], (80, 220, 80), f"throttle {row['throttle']:.2f}")
    bar(img, 14, 130, 270, 14, row["brake"], (60, 60, 230), f"brake {row['brake']:.2f}" + ("   (contact reflex)" if row.get("force_brake") else ""))
    tgt = row.get("target_kmh")
    put(img, f"speed  {row['speed_mps'] * 3.6:5.1f} km/h", 14, 180, 0.6)
    put(img, f"target {tgt:5.1f} km/h" if tgt is not None else "target   n/a", 14, 206, 0.6, (120, 120, 255))
    put(img, f"t {row['t']:.1f} s   tick {row['n']}", 14, 232, 0.5, DIM)
    if ended:
        put(img, "ROUTE ENDED - last frame held", 14, 262, 0.55, (0, 200, 255), 2)
    posterior(img, 300, 6, 300, 150, row)
    bev(img, 300, 162, 132, row, scale=3.2)
    put(img, "white = speed now, red = target", 440, 180, 0.42, DIM)
    experts(img, 620, 6, 330, row)
    return img


# ---------------------------------------------------------------- header + timeline
def header(name, result, scored, events_now, color):
    """`scored` is (label, result) of an earlier scored run of the same route, or empty."""
    img = np.full((H_HEAD, HALF, 3), (34, 34, 34), np.uint8)
    cv2.rectangle(img, (0, 0), (6, H_HEAD), color, -1)
    put(img, name, 16, 26, 0.7, TXT, 2)
    put(img, f"this drive: {result}" + (f"    {scored[0]}: {scored[1]}" if scored and scored[1] else ""), 16, 52, 0.5, DIM)
    if events_now:
        kind = events_now[-1]
        msg = "STOPPED > 60 s" if kind == "stuck" else f"COLLISION ({kind})"
        (tw, _), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
        cv2.rectangle(img, (HALF - tw - 30, 8), (HALF - 10, 40), EVENT_COLORS.get(kind, (60, 60, 255)), -1)
        put(img, msg, HALF - tw - 20, 32, 0.7, (255, 255, 255), 2)
    return img


def timeline_base(runs, events, names, t_max, width):
    img = np.full((H_TIME, width, 3), (18, 18, 18), np.uint8)
    x0, x1, y0, y1 = 60, width - 20, 14, H_TIME - 30
    v_max = 40.0
    for kmh in (0, 10, 20, 30, 40):
        y = int(y1 - (y1 - y0) * kmh / v_max)
        cv2.line(img, (x0, y), (x1, y), (45, 45, 45), 1)
        put(img, f"{kmh}", 26, y + 4, 0.4, DIM)
    put(img, "km/h", 8, y1 + 18, 0.4, DIM)
    for s in range(0, int(t_max) + 1, 30):
        x = int(x0 + (x1 - x0) * s / max(t_max, 1e-6))
        cv2.line(img, (x, y1), (x, y1 + 4), DIM, 1)
        put(img, f"{s}s", x - 10, y1 + 18, 0.4, DIM)

    def X(t):
        return int(x0 + (x1 - x0) * t / max(t_max, 1e-6))

    def Y(kmh):
        return int(y1 - (y1 - y0) * min(kmh, v_max) / v_max)

    for k, run in enumerate(runs):
        col = SIDE_COLORS[k]
        dim = tuple(int(c * 0.55) for c in col)
        rows = [run["rows"][n] for n in run["order"]]
        tg = [(X(r["t"] - rows[0]["t"]), Y(r["target_kmh"])) for r in rows if r.get("target_kmh") is not None]
        sp = [(X(r["t"] - rows[0]["t"]), Y(r["speed_mps"] * 3.6)) for r in rows]
        if len(tg) > 1:
            cv2.polylines(img, [np.int32(tg)], False, dim, 1, cv2.LINE_AA)
        if len(sp) > 1:
            cv2.polylines(img, [np.int32(sp)], False, col, 2, cv2.LINE_AA)
        for t, kind in events[k]:
            x = X(t - rows[0]["t"])
            yy = y0 + 4 + 12 * k
            cv2.drawMarker(img, (x, yy), EVENT_COLORS.get(kind, (60, 60, 255)), cv2.MARKER_TRIANGLE_DOWN, 12, 2)
    put(img, f"{names[0]}: speed (bright), target (dim)", x0 + 6, H_TIME - 6, 0.42, SIDE_COLORS[0])
    put(img, f"{names[1]}: speed (bright), target (dim)", x0 + 420, H_TIME - 6, 0.42, SIDE_COLORS[1])
    put(img, "triangles = collisions / long stops", x0 + 840, H_TIME - 6, 0.42, DIM)
    return img, (x0, x1, y0, y1)


# ---------------------------------------------------------------- compose
def compose(left_dir, right_dir, out, names, results, scored=((), ()), events=((), ()), fps=10.0, crf=30):
    runs = [load_run(left_dir), load_run(right_dir)]
    first = cv2.imread(os.path.join(left_dir, f"{runs[0]['order'][0]:06d}.jpg"))
    if first is None:
        raise SystemExit(f"no frame in {left_dir}")
    h_cam = int(round(HALF * first.shape[0] / first.shape[1] / 2)) * 2
    W, H = 2 * HALF, H_HEAD + h_cam + H_PANEL + H_TIME
    t0s = [run["rows"][run["order"][0]]["t"] for run in runs]
    t_max = max(run["rows"][run["order"][-1]]["t"] - t0 for run, t0 in zip(runs, t0s))
    evs = [parse_events(e) for e in events]
    tl_base, (tx0, tx1, ty0, ty1) = timeline_base(runs, evs, names, t_max, W)
    ticks = sorted(set(runs[0]["order"]) | set(runs[1]["order"]))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                          stdin=subprocess.PIPE)
    last = [None, None]
    for n in ticks:
        canvas = np.full((H, W, 3), BG, np.uint8)
        for k, run in enumerate(runs):
            row = run["rows"].get(n)
            ended = row is None and last[k] is not None and n > run["order"][-1]
            if row is not None:
                fr = cv2.imread(os.path.join(run["dir"], f"{n:06d}.jpg"))
                if fr is not None:
                    last[k] = (row, fr)
            if last[k] is None:
                continue
            row_k, fr_k = last[k]
            t_rel = row_k["t"]
            now = [kind for t, kind in evs[k] if t <= t_rel < t + 2.5]
            x = k * HALF
            canvas[0:H_HEAD, x:x + HALF] = header(names[k], results[k], scored[k], now, SIDE_COLORS[k])
            cam = camera_view(fr_k, row_k, run["meta"].get("layout"), HALF, h_cam)
            if ended:
                cam = (cam * 0.45).astype(np.uint8)
            canvas[H_HEAD:H_HEAD + h_cam, x:x + HALF] = cam
            canvas[H_HEAD + h_cam:H_HEAD + h_cam + H_PANEL, x:x + HALF] = panel(row_k, ended)
        cv2.line(canvas, (HALF, 0), (HALF, H_HEAD + h_cam + H_PANEL), (90, 90, 90), 2)
        tl = tl_base.copy()
        t_now = (n * runs[0]["meta"].get("tick_dt", 0.05))
        cx = int(tx0 + (tx1 - tx0) * min(t_now, t_max) / max(t_max, 1e-6))
        cv2.line(tl, (cx, ty0 - 6), (cx, ty1 + 4), (255, 255, 255), 1)
        canvas[H - H_TIME:H] = tl
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    return out, len(ticks)


# ---------------------------------------------------------------- batch
def _records(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        for r in json.load(open(f))["_checkpoint"]["records"]:
            rid = int(r["route_id"].split("_")[1])
            out[rid] = {"save_name": r.get("save_name"), "status": r["status"], "ds": float(r["scores"]["score_composed"]),
                        "scenario": r.get("scenario_name", "")}
    return out


def _fmt(r):
    return "not driven" if r is None else f"{r['status'].replace('Failed - ', '')}  DS {r['ds']:.1f}"


def _job(args):
    try:
        out, n = compose(*args)
        return f"OK   {os.path.basename(out)} ({n} frames)"
    except Exception as e:  # one bad route must not stop the batch
        return f"FAIL {os.path.basename(args[2])}: {e!r}"


def batch(a):
    L, R = _records(a.left_results), _records(a.right_results)
    S = _records(a.scored_right) if a.scored_right else {}
    SL = _records(a.scored_left) if a.scored_left else {}
    groups = json.load(open(a.groups)) if a.groups else {}
    jobs, index = [], []
    for rid in sorted(set(L) & set(R), key=lambda r: (groups.get(str(r), "z"), r)):
        ld = os.path.join(a.video_root, a.left_sub, L[rid]["save_name"] or "")
        rd = os.path.join(a.video_root, a.right_sub, R[rid]["save_name"] or "")
        if not (os.path.exists(os.path.join(ld, "ticks.jsonl")) and os.path.exists(os.path.join(rd, "ticks.jsonl"))):
            index.append({"route": rid, "video": None, "note": "recording missing"})
            continue
        grp = groups.get(str(rid), "route")
        scen = (R[rid]["scenario"] or "").split("_")[0]
        name = f"{grp}_{rid}_{scen}.mp4"
        ev = [sorted(glob.glob(os.path.join(a.clips_root, "*", f"{X[rid]['save_name']}_h*_*.npz"))) for X in (L, R)]
        scored = ((a.scored_labels[0], _fmt(SL.get(rid)) if SL else ""), (a.scored_labels[1], _fmt(S.get(rid)) if S else ""))
        if not (a.skip_existing and os.path.exists(os.path.join(a.out_dir, name))):
            jobs.append((ld, rd, os.path.join(a.out_dir, name), tuple(a.names), (_fmt(L[rid]), _fmt(R[rid])), scored, tuple(ev), a.fps, a.crf))
        index.append({"route": rid, "group": grp, "scenario": R[rid]["scenario"], "video": name,
                      "left": L[rid], "right": R[rid], "right_scored": S.get(rid), "left_scored": SL.get(rid),
                      "collisions_left": len(ev[0]), "collisions_right": len(ev[1])})
    os.makedirs(a.out_dir, exist_ok=True)
    with Pool(a.jobs) as p:
        for line in p.imap_unordered(_job, jobs):
            print(line, flush=True)
    json.dump(index, open(os.path.join(a.out_dir, "index.json"), "w"), indent=1)
    print(f"BATCH_DONE {len(jobs)} videos, index {os.path.join(a.out_dir, 'index.json')}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pair")
    p.add_argument("left_dir"); p.add_argument("right_dir"); p.add_argument("out")
    p.add_argument("--names", nargs=2, default=["left", "right"])
    p.add_argument("--results", nargs=2, default=["", ""])
    p.add_argument("--events-left", nargs="*", default=[])
    p.add_argument("--events-right", nargs="*", default=[])
    p.add_argument("--fps", type=float, default=10.0)
    b = sub.add_parser("batch")
    b.add_argument("--video-root", required=True); b.add_argument("--left-sub", required=True); b.add_argument("--right-sub", required=True)
    b.add_argument("--left-results", required=True); b.add_argument("--right-results", required=True)
    b.add_argument("--scored-left", default=""); b.add_argument("--scored-right", default="")
    b.add_argument("--clips-root", required=True); b.add_argument("--out-dir", required=True)
    b.add_argument("--groups", default=""); b.add_argument("--names", nargs=2, default=["left", "right"])
    b.add_argument("--scored-labels", nargs=2, default=["scored run", "scored run"])
    b.add_argument("--skip-existing", action="store_true", help="leave videos already in --out-dir (re-run as more pairs finish)")
    b.add_argument("--jobs", type=int, default=8); b.add_argument("--fps", type=float, default=10.0); b.add_argument("--crf", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "pair":
        out, n = compose(a.left_dir, a.right_dir, a.out, a.names, a.results, ((), ()), (a.events_left, a.events_right), a.fps)
        print(f"wrote {out} ({n} frames)")
    else:
        batch(a)


if __name__ == "__main__":
    sys.exit(main())
