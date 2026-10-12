"""TODO A44 (S-125): collision-clip recorder for the closed-loop evals (CARLA-free core; the agent glue is in scripts/eval/bench2drive_agent.py, flag B2D_CLIPS_DIR).

A ring buffer keeps the last `before_s` seconds at `hz` (default 6 s at 4 Hz) of: the downsized front frame (JPEG), the ego speed, the control, the speed-head posterior,
the target speed and the predicted waypoints. `trigger()` marks an event (vehicle / pedestrian / layout collision, or a long stand-still); `after_s` later the clip is
written to `<dir>/<tag>_<kind>_t<NNN>.npz` (~1 MB: 24 + 8 frames at 256 px wide, JPEG q70). Other actors' poses are stored in the event for analysis only: the policy never sees them.
Loading: `np.load(f, allow_pickle=True)` -> `jpegs` (object array of bytes), `scalars` (json), `event` (json).
"""
from __future__ import annotations

import io
import json
import os
from collections import deque
from typing import Dict, List, Optional

import numpy as np


def _jpeg(rgb: np.ndarray, width: int = 256, quality: int = 70) -> bytes:
    from PIL import Image
    im = Image.fromarray(np.ascontiguousarray(rgb))
    h = max(1, round(im.height * width / im.width))
    buf = io.BytesIO()
    im.resize((width, h)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class ClipRecorder:
    def __init__(self, out_dir: str, tag: str, hz: float = 4.0, before_s: float = 6.0, after_s: float = 2.0, tick_dt: float = 0.05, min_gap_s: float = 20.0,
                 max_per_kind: int = 6):
        # min_gap_s per (kind, other actor): a car in continuous contact fires the collision sensor every tick (the S-126 test: 31 clips in 155 s of pushing a parked car);
        # max_per_kind caps what one route can write
        self.dir, self.tag, self.after_s, self.min_gap_s, self.max_per_kind = out_dir, tag, after_s, min_gap_s, max_per_kind
        self.count: Dict[str, int] = {}
        self.every = max(1, round(1.0 / (hz * tick_dt)))
        self.buf: deque = deque(maxlen=max(2, int(before_s * hz)))
        self.pending: List[Dict] = []
        self.n = 0
        self.last_trigger: Dict[str, float] = {}
        self.written: List[str] = []
        os.makedirs(out_dir, exist_ok=True)

    def push(self, t: float, rgb: Optional[np.ndarray], scalars: Dict) -> None:
        """Called every tick; stores every `every`-th tick, and finishes the pending clips whose after-window has passed."""
        self.n += 1
        if self.n % self.every == 0 and rgb is not None:
            self.buf.append((t, _jpeg(rgb), scalars))
            for p in self.pending:
                if t <= p["t"] + self.after_s:
                    p["after"].append(self.buf[-1])
        for p in [p for p in self.pending if t >= p["t"] + self.after_s]:
            self._write(p)
            self.pending.remove(p)

    def trigger(self, t: float, kind: str, event: Dict) -> bool:
        key = f"{kind}:{event.get('other_id', '')}"
        if t - self.last_trigger.get(key, -1e9) < self.min_gap_s or self.count.get(kind, 0) >= self.max_per_kind:
            return False
        self.last_trigger[key] = t
        self.count[kind] = self.count.get(kind, 0) + 1
        self.pending.append({"t": t, "kind": kind, "event": event, "before": list(self.buf), "after": []})
        return True

    def close(self) -> None:
        for p in self.pending:
            self._write(p)
        self.pending = []

    def _write(self, p: Dict) -> None:
        frames = p["before"] + p["after"]
        path = os.path.join(self.dir, f"{self.tag}_{p['kind']}_t{int(p['t']):04d}.npz")
        jp = np.empty(len(frames), dtype=object)
        for i, f in enumerate(frames):
            jp[i] = f[1]
        np.savez_compressed(path, jpegs=jp, scalars=json.dumps([{"t": f[0], **f[2]} for f in frames], default=float),
                            event=json.dumps({"t": p["t"], "kind": p["kind"], **p["event"]}, default=float), n_before=len(p["before"]))
        self.written.append(path)
