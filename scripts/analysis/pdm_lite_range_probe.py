#!/usr/bin/env python3
"""Read a few files out of a PDM-Lite zip on the Hugging Face Hub with HTTP range requests (no archive download).

The dataset (autonomousvision/PDM_Lite_Carla_LB2) ships one multi-GB zip per scenario and town. Listing a zip's central directory and
pulling single `measurements/*.json.gz` / `boxes/*.json.gz` files costs a few MB, enough to check which fields the logs carry
(TODO A21 / A37 / A42 / A45) before any dataset box is rented.

    py scripts/analysis/pdm_lite_range_probe.py Town12/data/HardBreakRoute.zip [--routes 2] [--frames 6]
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import zipfile

import requests

BASE = "https://huggingface.co/datasets/autonomousvision/PDM_Lite_Carla_LB2/resolve/main/"


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url, self.pos = url, 0
        r = requests.head(url, allow_redirects=True, timeout=60)
        self.size = int(r.headers["Content-Length"])
        self.url = r.url

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        r = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        r.raise_for_status()
        self.pos += len(r.content)
        return r.content


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip_path", help="e.g. Town12/data/HardBreakRoute.zip")
    ap.add_argument("--routes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--out", default=None, help="write the sampled measurements/boxes here as JSON")
    a = ap.parse_args()
    zf = zipfile.ZipFile(HttpRangeFile(BASE + a.zip_path))
    names = zf.namelist()
    print(len(names), "entries in", a.zip_path)
    routes = sorted({m.group(1) for n in names if (m := re.match(r"(.+?/Route\d+_Rep\d+)/", n))})
    print(len(routes), "routes; first:", routes[:3])
    sample = {}
    for route in routes[: a.routes]:
        meas = sorted(n for n in names if n.startswith(route + "/measurements/") and n.endswith(".json.gz"))
        boxes = sorted(n for n in names if n.startswith(route + "/boxes/") and n.endswith(".json.gz"))
        print(f"{route}: {len(meas)} measurement files, {len(boxes)} box files, "
              f"sensors: {sorted({n.split('/')[1] for n in names if n.startswith(route + '/')})}")
        step = max(1, len(meas) // a.frames)
        for n in meas[::step][: a.frames]:
            m = json.loads(gzip.decompress(zf.read(n)))
            b = json.loads(gzip.decompress(zf.read(n.replace("measurements", "boxes"))))
            sample[n] = {"measurements": m, "boxes": b}
    if a.out:
        json.dump(sample, open(a.out, "w"))
    first = next(iter(sample.values()))
    print("measurement keys:", sorted(first["measurements"].keys()))
    box = first["boxes"][0] if first["boxes"] else {}
    print("box keys:", sorted(box.keys()), "| n boxes in frame:", len(first["boxes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
