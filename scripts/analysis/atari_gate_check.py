"""Atari Gate 1 acceptance check (2026-11-03): reads a trainer log (train_ez_offpolicy.py) and returns the gate verdict.
Gate 1 (brief): attention entropy above 0.8 (mean over layers, normalised: 1 = uniform), no collapse of action selection to more than 85%
NOOP + RIGHT, finite losses. A run with no attention line fails: an unmeasured gate is not a passed gate.
usage: py scripts/analysis/atari_gate_check.py LOG [LOG ...]"""
import math
import re
import sys

GATE = {"attn_min": 0.8, "noop_right_max": 0.85}
_ATTN = re.compile(r"attn H ([0-9.]+(?:/[0-9.]+)*) pol ([0-9.]+)")
_LOSS = re.compile(r"loss v (\S+) p (\S+) r (\S+)")
_SHARE = re.compile(r"act share ([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)")


def parse_log_line(line: str) -> dict:
    out = {"attn_layers": None, "attn_policy": None, "loss_v": None, "loss_p": None, "act_share": None}
    m = _ATTN.search(line)
    if m:
        out["attn_layers"] = [float(x) for x in m.group(1).split("/")]
        out["attn_policy"] = float(m.group(2))
    m = _LOSS.search(line)
    if m:
        out["loss_v"], out["loss_p"] = float(m.group(1)), float(m.group(2))
    m = _SHARE.search(line)
    if m:
        out["act_share"] = [float(m.group(i)) for i in range(1, 5)]
    return out


def gate_verdict(lines) -> tuple:
    parsed = [parse_log_line(l) for l in lines]
    reasons = []
    attn = [p["attn_layers"] for p in parsed if p["attn_layers"] is not None]
    if not attn:
        reasons.append("no attention entropy line: the gate cannot be checked")
    else:
        mean_h = sum(sum(a) / len(a) for a in attn) / len(attn)
        if mean_h <= GATE["attn_min"]:
            reasons.append(f"attention entropy {mean_h:.2f} <= {GATE['attn_min']} (collapsing)")
    for p in parsed:
        if p["loss_v"] is not None and not all(math.isfinite(x) for x in (p["loss_v"], p["loss_p"])):
            reasons.append("non-finite loss")
            break
    shares = [p["act_share"] for p in parsed if p["act_share"] is not None]
    if shares and max(s[0] + s[2] for s in shares) > GATE["noop_right_max"]:
        reasons.append("action selection collapsed to more than 85% NOOP/RIGHT")
    return (not reasons), reasons


if __name__ == "__main__":
    failed = False
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8", errors="replace") as f:
            ok, reasons = gate_verdict(f.read().splitlines())
        failed = failed or not ok
        print(("PASS " if ok else "FAIL ") + path + ("" if ok else ": " + "; ".join(reasons)))
    sys.exit(1 if failed else 0)
