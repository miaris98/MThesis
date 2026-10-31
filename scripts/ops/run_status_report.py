#!/usr/bin/env python3
"""The 10-minute run report (user rule, 2026-10-28): errors, progress with the last log lines, checkpoint state and ETA.

One block per run: errors (count and the last error line), progress (env / update, return, ETA in Athens time), whether the checkpoint and
replay pair is saved and how old it is, the last E: backup line, and the last two log lines. A run on a box is read over SSH (--ssh
root@HOST --port P, paths are on the box); a run on this PC is read from local paths (no --ssh). Schedule it every 10 minutes.
From Git Bash on Windows, set MSYS_NO_PATHCONV=1 for --ssh: otherwise the box paths (/workspace/...) arrive as C:/Program Files/Git/workspace/...
and every box-side read comes back empty (no progress, no checkpoints; found 2026-10-28).

usage:
  python scripts/ops/run_status_report.py --label a_plain_v2 --log LOG --run RUN_DIR --total-upd 960
  python scripts/ops/run_status_report.py --ssh root@HOST --port P --label b66_sgd_s1 \\
      --log /workspace/b66/logs/b66_sgd_s1.log --run /workspace/b66/out/b66_sgd_s1 \\
      --save-log /e/MThesis_EXP/live_20261026_b66/save.log --gpus --credit
"""
import argparse
import datetime as dt
import re
import subprocess
from pathlib import Path

ERR = r"Traceback|Error|out of memory|Killed|exited rc=[1-9]|non-finite"
PROG = re.compile(r"^env\s+(\d+)/(\d+)\s+upd\s+(\d+)/(\d+)\s.*?train-life-ret\(20\)\s+([-\d.]+).*?([\d.]+) min\s*$")
CKPT = re.compile(r"^checkpoint_env(\d+)_upd(\d+)\.pt$")
PAIR_TOLERANCE_S = 180   # checkpoint_latest.pt and replay_latest.npz are written in one save, so their mtimes sit this close
VASTAI = r"C:\Users\miari\anaconda3\envs\graphtools\Scripts\vastai.exe"


class Where:
    """This PC (ssh None) or a box over ssh. Every method returns plain text or values; a failed ssh call reads as unreachable."""

    def __init__(self, ssh=None, port=None):
        self.ssh, self.port = ssh, port

    def run(self, cmd):
        r = subprocess.run(["ssh", "-p", str(self.port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.ssh, cmd],
                           capture_output=True, text=True, timeout=120)
        return r.stdout if r.returncode == 0 else None

    def reachable(self):
        return True if self.ssh is None else self.run("echo ok") is not None

    def tail(self, path, n=400):
        if self.ssh is None:
            return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
        return (self.run(f"tail -n {n} '{path}'") or "").splitlines()

    def errors(self, path):
        """(count, last matching line) over the whole log."""
        if self.ssh is None:
            hits = [x for x in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if re.search(ERR, x)]
        else:
            hits = (self.run(f"grep -E '{ERR}' '{path}'") or "").splitlines()
        return len(hits), (hits[-1][:160] if hits else "")

    def log_age_min(self, path):
        t = self.mtime(path)
        return None if t is None else round((time_now() - t) / 60, 1)

    def mtime(self, path):
        if self.ssh is None:
            p = Path(path)
            return int(p.stat().st_mtime) if p.exists() else None
        out = (self.run(f"stat -c %Y '{path}' 2>/dev/null") or "").strip()
        return int(out) if out.isdigit() else None

    def listing(self, folder):
        """{file name: (size bytes, mtime epoch)} of a folder."""
        if self.ssh is None:
            p = Path(folder)
            if not p.is_dir():
                return {}
            return {f.name: (f.stat().st_size, int(f.stat().st_mtime)) for f in p.iterdir() if f.is_file()}
        res = {}
        for line in (self.run(f"ls -l --time-style=+%s '{folder}' 2>/dev/null") or "").splitlines():
            parts = line.split()
            if len(parts) == 7 and parts[0].startswith("-"):      # perms links owner group size mtime name
                res[parts[6]] = (int(parts[4]), int(parts[5]))
        return res

    def alive(self, label):
        """Trainer processes for this run label on the box ('[r]un-label' keeps this command's own line from matching)."""
        if self.ssh is None:
            return None
        out = (self.run(f"pgrep -fc '[r]un-label {label}( |$)' 2>/dev/null") or "").strip()
        return int(out) if out.isdigit() else 0

    def gpus(self):
        out = (self.run("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits") or "")
        procs = (self.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader") or "").split()
        return [g.strip() for g in out.splitlines() if g.strip()], len(procs)


def time_now():
    return dt.datetime.now().timestamp()


def athens(ts_epoch=None):
    t = dt.datetime.now() if ts_epoch is None else dt.datetime.fromtimestamp(ts_epoch)
    return t.strftime("%H:%M")


def report_run(w, label, log, run, total_upd, save_log):
    ok = w.reachable()
    head = f"== {athens()} Athens | {label} | {'box' if w.ssh else 'this PC'}"
    if not ok:
        return [head + " | UNREACHABLE (no ssh answer; last known state is in the E: backup line below)"]
    n_err, last_err = w.errors(log)
    alive = w.alive(label)
    lines = [head]
    age = w.log_age_min(log)
    lines.append(f"errors: {n_err}" + (f" (last: {last_err})" if n_err else "") +
                 (f" | trainer processes: {alive}" if alive is not None else "") +
                 (f" | log written {age} min ago" if age is not None else ""))
    prog = None
    for x in w.tail(log, 3000):
        m = PROG.match(x)
        if m:
            prog = m
    ck = w.listing(f"{run}/checkpoints")
    named = sorted(((int(m.group(2)), int(m.group(1)), n) for n in ck for m in [CKPT.match(n)] if m), reverse=True)
    if prog:
        env, env_tot, upd, upd_tot, ret, minutes = prog.groups()
        upd, upd_tot, minutes = int(upd), int(upd_tot), float(minutes)
        rate = upd / minutes if minutes > 0 else 0
        left = (upd_tot - upd) / rate if rate > 0 else None
        eta = "n/a" if left is None or upd < 500 else (dt.datetime.now() + dt.timedelta(minutes=left)).strftime("%H:%M") + f" ({left / 60:.1f} h left)"
        lines.append(f"progress: env {env}/{env_tot}, update {upd}/{upd_tot} ({100 * upd / upd_tot:.0f}%), return(20) {ret}, ETA {eta}")
    else:
        upd_tot = total_upd
        lines.append(f"progress: no progress line yet (the trainer prints every 1,000 env steps)" +
                     (f"; ETA n/a, total {total_upd} updates" if total_upd else ""))
    if named:
        upd, env, name = named[0]
        mt = ck.get("checkpoint_latest.pt", (0, None))[1]
        rp = ck.get("replay_latest.npz")
        pair = "n/a"
        if mt and rp:
            gap = abs(mt - rp[1])
            pair = f"replay pair {'OK' if gap <= PAIR_TOLERANCE_S else f'NOT matched ({gap} s apart)'} ({rp[0] / 1e9:.2f} GB)"
        latest_age = round((time_now() - mt) / 60, 1) if mt else None
        lines.append(f"checkpoints: newest {name} (update {upd}), latest saved {latest_age} min ago, {pair}")
    else:
        lines.append("checkpoints: none saved yet")
    if save_log:
        bl = [x for x in Path(save_log).read_text(encoding="utf-8", errors="replace").splitlines() if label in x] if Path(save_log).exists() else []
        lines.append(f"E: backup (UTC): {bl[-1][:140] if bl else 'no line for this run yet'}")
    lines.append("log:")
    for x in w.tail(log, 2)[-2:]:
        lines.append("  " + x[:160])
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh", help="user@host for a box; omit for runs on this PC")
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--label", action="append", required=True, help="repeat for several runs (same order as --log / --run)")
    ap.add_argument("--log", action="append", required=True)
    ap.add_argument("--run", action="append", required=True, help="the run folder (its checkpoints/ holds the saves)")
    ap.add_argument("--total-upd", type=int, default=None, help="updates of the whole run, when no progress line exists yet")
    ap.add_argument("--save-log", default=None, help="the PC's E: backup log (its lines carry the run label)")
    ap.add_argument("--gpus", action="store_true", help="box only: GPU utilisation and memory, and the number of GPU processes")
    ap.add_argument("--credit", action="store_true", help="Vast credit, through the vastai CLI (it reads its own login)")
    a = ap.parse_args()
    if not (len(a.label) == len(a.log) == len(a.run)):
        raise SystemExit("--label, --log and --run must be given the same number of times")
    w = Where(a.ssh, a.port)
    for label, log, run in zip(a.label, a.log, a.run):
        print("\n".join(report_run(w, label, log, run, a.total_upd, a.save_log)))
    if a.gpus and a.ssh and w.reachable():
        g, procs = w.gpus()
        print("gpus (index, name, util %, mem MiB used, mem MiB total): " + " | ".join(g) + f" | GPU processes: {procs}")
    if a.credit:
        try:
            out = subprocess.run([VASTAI, "show", "user", "--raw"], capture_output=True, text=True, timeout=120).stdout
        except (OSError, subprocess.TimeoutExpired):
            out = ""
        m = re.search(r'"credit":\s*([-\d.]+)', out or "")
        print(f"credit: ${float(m.group(1)):.2f}" if m else "credit: unknown (vastai CLI gave no answer)")


if __name__ == "__main__":
    main()
