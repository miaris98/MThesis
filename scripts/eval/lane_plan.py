#!/usr/bin/env python3
"""Route bookkeeping for one CARLA lane job (TODO A60; the 2026-10-19 incidents 1 and 2). Called by queue_A9c.sh's eval_job.

    python3 lane_plan.py plan LABEL ROUTES OUT_DIR   -> "LABEL ROUTES": what to launch ("LABEL -" if nothing is left to drive)
    python3 lane_plan.py split ROUTES PROGRESS [LOG] -> "ROUTE REST": the crash route and the routes after it; the route the attempt
                                                        log LOG was on when given, else the route at PROGRESS
    python3 lane_plan.py norm ROUTES                 -> ROUTES in ascending order, duplicates removed

1. Run order. The evaluator drives a route list in ascending id order whatever order it was given in, but the crash continuation
   indexed the list as given: with an unsorted list, a crash route at its start blocked or skipped the wrong routes (2026-10-19,
   C2 chain l1x: no record for 75 min). Every list is normalised to ascending order here, so list position = run position.
2. Changed lists. Relaunching a label on a different route list makes the evaluator's resume reset the label's result file
   (2026-10-19, C1 lane 1: about 36 records lost, and the hourly save then overwrote the E: copy). `plan` never relaunches a label
   on another list. A label is resumed only when its stored list (`<OUT>/<label>_x.routes`) is the requested set; otherwise the
   derived labels `<label>d1`, `<label>d2`, ... are tried in turn, each resumed only when its stored list is exactly what was still
   left when it was created, and else a new derived label gets the requested routes nobody has driven yet. A route counts as
   driven when it has a driving record; a harness failure ("couldn't be set up", "Simulation crashed", "Agent crashed") does not.
   The derived labels keep the arm's label prefix, so a9_merge.py / a3_failure_classes.py pool them with the original.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Iterable, List, Optional, Set, Tuple

INFRA = ("couldn't be set up", "Simulation crashed", "Agent crashed")
MAX_DERIVED = 50


def parse(routes: str) -> List[int]:
    return sorted({int(x) for x in routes.replace(" ", "").replace("\n", ",").split(",") if x})


def fmt(ids: Iterable[int]) -> str:
    return ",".join(map(str, ids))


def _checkpoint(result_json: str) -> dict:
    try:
        with open(result_json, encoding="utf-8") as fh:
            return json.load(fh).get("_checkpoint", {}) or {}
    except (OSError, ValueError, AttributeError):
        return {}


def _route_ids(result_json: str, driving_only: bool) -> Set[int]:
    out = set()
    for rec in _checkpoint(result_json).get("records", []) or []:
        if driving_only and any(k in str(rec.get("status", "")) for k in INFRA):
            continue
        try:
            out.add(int(str(rec["route_id"]).split("_")[1]))
        except (KeyError, IndexError, ValueError):
            continue
    return out


def driven(result_json: str) -> Set[int]:
    """Route ids with a driving record in a Leaderboard result file (empty if the file is missing or unreadable)."""
    return _route_ids(result_json, driving_only=True)


def recorded(result_json: str) -> Set[int]:
    """Route ids with any record, harness failures included."""
    return _route_ids(result_json, driving_only=False)


def stored(out_dir: str, label: str) -> Optional[List[int]]:
    path = os.path.join(out_dir, f"{label}_x.routes")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return parse(fh.read())


def plan(label: str, routes: str, out_dir: str) -> Tuple[str, List[int]]:
    want = parse(routes)
    res = os.path.join(out_dir, f"{label}_x.json")
    mine = stored(out_dir, label)
    if mine is None:
        total = (_checkpoint(res).get("progress") or [0, 0])[-1] if os.path.exists(res) else 0
        if not total or (total == len(want) and recorded(res) <= set(want)):
            return label, want                      # a new label, or one from before route files existed, on a list it fits
    elif set(mine) == set(want):
        return label, want                          # the same list: resume
    done = driven(res)
    for k in range(1, MAX_DERIVED + 1):
        cand = f"{label}d{k}"
        left = [r for r in want if r not in done]
        prev = stored(out_dir, cand)
        if prev is None and not os.path.exists(os.path.join(out_dir, f"{cand}_x.json")):
            return cand, left                       # first unused derived label: the routes nobody has driven yet
        if prev is not None and set(prev) == set(left):
            return cand, left                       # created for exactly this remainder: resume it
        done |= driven(os.path.join(out_dir, f"{cand}_x.json"))
    raise SystemExit(f"lane_plan: more than {MAX_DERIVED} derived labels for {label}")


PREPARING = re.compile(r"Preparing RouteScenario_(\d+)")


def last_prepared(attempt_log: Optional[str]) -> Optional[int]:
    """The route the evaluator was on when it stopped: the last 'Preparing RouteScenario_<id>' line of an attempt log."""
    if not attempt_log or not os.path.exists(attempt_log):
        return None
    with open(attempt_log, encoding="utf-8", errors="replace") as fh:
        hits = PREPARING.findall(fh.read())
    return int(hits[-1]) if hits else None


def split(routes: str, progress: int, attempt_log: Optional[str] = None) -> Tuple[str, str]:
    """The crash route and the routes after it.

    A CARLA segfault while loading leaves no record, so the crash route is the one at PROGRESS. A crash during the route
    ('Failed - Simulation crashed', e.g. the scenario runner's 'trying to operate on a destroyed actor') writes a record first, so
    PROGRESS already counts it and ids[progress] is a healthy route (2026-11-18: 3572 and 2086 excluded, 3520 and 1956 crashed).
    The attempt log names the route either way, so it wins when it names a route of this list."""
    ids = parse(routes)
    on = last_prepared(attempt_log)
    if on is not None and on in ids:
        k = ids.index(on)
        return str(on), fmt(ids[k + 1:])
    if progress >= len(ids):
        return "", ""
    return str(ids[progress]), fmt(ids[progress + 1:])


def main(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[0] == "norm":
        print(fmt(parse(argv[1])))
    elif len(argv) == 4 and argv[0] == "plan":
        lab, left = plan(argv[1], argv[2], argv[3])
        print(lab, fmt(left) if left else "-")
    elif len(argv) in (3, 4) and argv[0] == "split":
        print(*split(argv[1], int(argv[2]), argv[3] if len(argv) == 4 else None))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
