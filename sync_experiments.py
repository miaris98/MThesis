#!/usr/bin/env python3
"""Pull vast.ai experiment output onto the external disk and merge it into one MLflow store.

Two halves, usable together or separately:

  pull    tar-over-ssh the remote /workspace output into <EXP_ROOT>/imports/<tag>/.
          One stream instead of thousands of scp round-trips - an MLflow file store
          is mostly tiny files, which scp -r handles very badly.

  merge   fold an imported mlruns tree into <EXP_ROOT>/mlruns so local and remote
          runs show up in a single MLflow UI. Experiment *ids* are sequential
          integers and collide across machines, so experiments are matched by
          NAME and remapped onto a locally-allocated id; every run's artifact_uri
          is rewritten to its new home, and each imported run is tagged with the
          instance it came from.

Also:

  relink  rewrite every artifact_uri/artifact_location in the local store to match
          where the store actually sits. Run this if the external disk comes back
          on a different drive letter, otherwise MLflow serves broken artifacts.

Stdlib only, on purpose: this has to work before mlflow is installed locally.

Examples
--------
    # Pull metrics + telemetry (fast) and merge into the local store
    python sync_experiments.py --ssh-cmd "ssh -p 12345 root@ssh5.vast.ai"

    # Everything, including checkpoints, TensorBoard events and videos
    python sync_experiments.py --ssh-cmd "..." --include all

    # Merge a snapshot pulled earlier, without touching the network
    python sync_experiments.py --merge-only "E:/MThesis_EXP/imports/ssh5-12345_20260906-1200"

    # Repair artifact paths after the disk changed drive letter
    python sync_experiments.py --relink
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: How often the transfer progress line refreshes, in seconds. The tar stream gives no
#: total size up front (the remote side only knows once it's done), so this reports
#: elapsed time, bytes received so far, and instantaneous rate rather than a percentage
#: - the only honest options without a size the remote could report before finishing.
PROGRESS_INTERVAL_S = 2.0

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import paths  # noqa: E402


#: Remote paths, relative to --remote-root, grouped so a metrics-only pull stays fast.
#: The mlruns location matches the backend store run_multi_carla_training.sh serves.
INCLUDE_GROUPS: Dict[str, List[str]] = {
    "mlruns": ["MThesis/mlruns"],
    "telemetry": ["runs/training_telemetry.csv", "MThesis/wor_training_telemetry.csv"],
    "tensorboard": ["runs"],
    "checkpoints": ["checkpoints"],
    "videos": ["eval_video.mp4", "eval_video_best.mp4", "wor_eval_video.mp4"],
    # The actual experimental record behind every closed-loop/Bench2Drive table in
    # challenges_13.md and TODO_leaderboard_benchmark.md: driving-score JSON, per-route
    # logs and their rollout videos. Not covered by "videos" above (that group predates
    # this output layout) or "checkpoints" (a sibling tree, not nested under it). Each
    # entry is a whole directory, since build_remote_tar_cmd single-quotes every path
    # (globs like train_*.log would not expand under that quoting, so logs aren't in
    # this group - pull them by hand if needed, e.g. scp 'root@host:/workspace/train_*.log').
    "results": ["closed_loop", "closed_loop_official", "closed_loop_seed1",
                "closed_loop_seed2", "bench2drive_out"],
}
#: What a bare invocation pulls: small, and enough to compare runs.
DEFAULT_INCLUDES = ["mlruns", "telemetry"]

#: Directories inside an MLflow file store that are not experiments.
RESERVED_STORE_DIRS = {".trash", "models", ".mlflow"}

#: Directories inside an *experiment* that are not runs. Mirrors MLflow's own
#: FileStore.RESERVED_EXPERIMENT_FOLDERS - experiment-level tags, logged datasets,
#: traces and registered models all live beside the run directories.
RESERVED_EXPERIMENT_DIRS = {"tags", "datasets", "traces", "models"}


# --------------------------------------------------------------------------- #
# Minimal flat-YAML IO. MLflow's file store writes one level of scalar keys, so
# a full YAML parser is not worth a hard dependency here.
# --------------------------------------------------------------------------- #

def read_meta(path: Path) -> Dict[str, str]:
    """Parse an MLflow meta.yaml into a flat dict of strings."""
    out: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip() or line[:1] in (" ", "\t", "-", "#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key.strip()] = value
    return out


def _scalar(value) -> str:
    """Render a value as a quoted YAML scalar (or a bare literal where required)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def update_meta(path: Path, updates: Dict[str, object]) -> None:
    """Rewrite the given top-level keys in a meta.yaml, appending any that are absent."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    seen = set()
    out = []
    for line in lines:
        key = line.partition(":")[0].strip()
        if key in updates and line[:1] not in (" ", "\t", "-"):
            out.append(key + ": " + _scalar(updates[key]))
            seen.add(key)
        else:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(key + ": " + _scalar(value))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Pull
# --------------------------------------------------------------------------- #

def parse_ssh_cmd(ssh_cmd: str) -> Tuple[str, str]:
    """Extract (port, user@host) from a vast.ai console string."""
    port_match = re.search(r"-p\s+(\d+)", ssh_cmd)
    port = port_match.group(1) if port_match else "22"
    host_match = re.search(r"([A-Za-z0-9_.\-]+@[A-Za-z0-9.\-]+)", ssh_cmd)
    if host_match:
        return port, host_match.group(1)
    parts = [p for p in ssh_cmd.split() if not p.startswith("-") and p not in ("ssh", port)]
    return port, ("root@" + parts[-1] if parts else "root@localhost")


def resolve_includes(names: List[str]) -> List[str]:
    """Expand include group names into remote-relative paths."""
    if "all" in names:
        names = list(INCLUDE_GROUPS)
    selected: List[str] = []
    for name in names:
        if name not in INCLUDE_GROUPS:
            raise SystemExit(
                "Unknown --include group '%s'. Choose from: all, %s"
                % (name, ", ".join(INCLUDE_GROUPS))
            )
        for path in INCLUDE_GROUPS[name]:
            if path not in selected:
                selected.append(path)
    return selected


def build_remote_tar_cmd(remote_root: str, rel_paths: List[str]) -> str:
    """Remote shell snippet that tars only the paths that actually exist.

    Filtering on the remote side keeps a missing checkpoint or video from failing
    the whole transfer, which is the normal case mid-run. Exit 9 = bad root,
    exit 8 = nothing to send; both are reported as advice rather than a stack trace.
    """
    quoted = " ".join("'" + p + "'" for p in rel_paths)
    return (
        "cd '" + remote_root + "' 2>/dev/null || exit 9; "
        "L=''; for p in " + quoted + '; do [ -e "$p" ] && L="$L $p"; done; '
        'if [ -z "$L" ]; then exit 8; fi; '
        "tar -czf - $L 2>/dev/null"
    )


def extract_snapshot(archive: Path, dest: Path) -> None:
    """Unpack a pulled tar.gz into ``dest``."""
    with tarfile.open(archive, "r:gz") as tar:
        # data_filter landed in 3.12 and becomes the default in 3.14; pass it
        # where available and fall back cleanly on older interpreters.
        try:
            tar.extractall(dest, filter="data")
        except TypeError:
            tar.extractall(dest)


def _format_eta(elapsed: float) -> str:
    m, s = divmod(int(elapsed), 60)
    return "%dm%02ds" % (m, s) if m else "%ds" % s


def _run_with_progress(cmd: List[str], sink_path: Path) -> int:
    """Runs `cmd`, streaming its stdout into `sink_path`, printing a live progress
    line while it does.

    subprocess.run(..., stdout=sink) blocks silently until the whole transfer
    finishes, which for a multi-run checkpoint pull can be minutes with zero
    feedback - indistinguishable from a hang. Popen + polling the growing output
    file's size is what lets a line update in place instead.
    """
    is_tty = sys.stdout.isatty()
    with open(sink_path, "wb") as sink:
        proc = subprocess.Popen(cmd, stdout=sink)
        started = time.time()
        last_size = 0
        last_tick = started
        try:
            while proc.poll() is None:
                time.sleep(PROGRESS_INTERVAL_S)
                now = time.time()
                size = sink_path.stat().st_size
                rate = (size - last_size) / max(1e-6, now - last_tick)
                line = "    ...received %6.1f MB | %5.1f MB/s | elapsed %s" % (
                    size / 1e6, rate / 1e6, _format_eta(now - started)
                )
                if is_tty:
                    # \r overwrites the line in place; pad so a shorter line fully
                    # covers a longer previous one instead of leaving stray tail text.
                    print("\r" + line.ljust(70), end="", flush=True)
                else:
                    # No terminal to overwrite (piped output, CI, this session's tool
                    # capture) - one line per tick instead, still far better than
                    # nothing until the transfer completes.
                    print(line, flush=True)
                last_size, last_tick = size, now
        finally:
            if is_tty:
                print()  # leave the final progress line intact, move to a fresh one
        proc.wait()
        return proc.returncode


def pull(port: str, user_host: str, remote_root: str, includes: List[str], dest: Path,
         ssh_opts: Optional[List[str]] = None, ssh_bin: str = "ssh") -> Path:
    """Stream the selected remote output into ``dest`` and return that directory."""
    rel_paths = resolve_includes(includes)
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [ssh_bin, "-p", str(port)]
    cmd += ssh_opts or ["-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]
    cmd += [user_host, build_remote_tar_cmd(remote_root, rel_paths)]

    print("--> Pulling %s from %s:%s" % (", ".join(includes), user_host, remote_root))
    print("    Paths: " + ", ".join(rel_paths))

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz", prefix="mthesis_sync_")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    started = time.time()
    try:
        returncode = _run_with_progress(cmd, tmp_path)
        if returncode == 9:
            raise SystemExit("Remote root '%s' does not exist on the instance." % remote_root)
        if returncode == 8:
            raise SystemExit("None of the requested paths exist on the instance yet - nothing to pull.")
        size = tmp_path.stat().st_size
        if returncode != 0 or size == 0:
            raise SystemExit(
                "Transfer failed (ssh exit %s, %d bytes received)." % (returncode, size)
            )

        print("    Received %.1f MB in %s, extracting..." % (size / 1e6, _format_eta(time.time() - started)))
        extract_snapshot(tmp_path, dest)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    print("[ok] Snapshot at " + str(dest))
    return dest


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def find_mlruns(root: Path) -> Optional[Path]:
    """Locate the mlruns directory inside a pulled snapshot."""
    if root.name == "mlruns" and root.is_dir():
        return root
    for candidate in (root / "mlruns", root / "MThesis" / "mlruns"):
        if candidate.is_dir():
            return candidate
    for candidate in sorted(root.rglob("mlruns")):
        if candidate.is_dir():
            return candidate
    return None


def experiment_dirs(store: Path) -> List[Path]:
    """Experiment directories in an MLflow file store, skipping bookkeeping dirs."""
    if not store.is_dir():
        return []
    return sorted(
        d for d in store.iterdir()
        if d.is_dir() and d.name not in RESERVED_STORE_DIRS and (d / "meta.yaml").is_file()
    )


def run_dirs(experiment: Path) -> List[Path]:
    """Run directories inside an experiment, excluding MLflow's reserved folders."""
    return sorted(
        d for d in experiment.iterdir()
        if d.is_dir() and d.name not in RESERVED_EXPERIMENT_DIRS
    )


def experiments_by_name(store: Path) -> Dict[str, Path]:
    """Map experiment name -> its directory in ``store``."""
    mapping: Dict[str, Path] = {}
    for d in experiment_dirs(store):
        name = read_meta(d / "meta.yaml").get("name") or d.name
        mapping[name] = d
    return mapping


def allocate_experiment_id(store: Path, preferred: Optional[str] = None) -> str:
    """Pick a free experiment id in ``store``, keeping ``preferred`` when it is free.

    MLflow 2.x numbered experiments 0, 1, 2..., which collides across machines;
    3.x assigns large random integers, which effectively never does. Keeping the
    source id where possible means a run keeps the same coordinates it had on the
    instance, and only genuine collisions get renumbered.
    """
    used = set()
    if store.is_dir():
        for d in store.iterdir():
            if d.is_dir() and d.name.isdigit():
                used.add(int(d.name))
    if preferred and preferred.isdigit() and int(preferred) not in used:
        return preferred
    return str(max(used) + 1) if used else "0"


def write_run_tag(run_dir: Path, key: str, value: str) -> None:
    """Set an MLflow tag by writing the file store's one-file-per-tag representation."""
    tags = run_dir / "tags"
    tags.mkdir(parents=True, exist_ok=True)
    (tags / key).write_text(str(value), encoding="utf-8")


def merge_store(src_store: Path, dst_store: Path, source_label: str,
                overwrite: bool = False, verbose: bool = True) -> Tuple[int, int]:
    """Merge ``src_store`` into ``dst_store``. Returns (runs merged, runs skipped)."""
    dst_store.mkdir(parents=True, exist_ok=True)
    dst_by_name = experiments_by_name(dst_store)
    merged = skipped = 0

    for src_exp in experiment_dirs(src_store):
        src_meta = read_meta(src_exp / "meta.yaml")
        name = src_meta.get("name") or src_exp.name

        # Name is the identity that has to survive the move: MLflow requires unique
        # experiment names, so an existing local experiment of the same name is the
        # one to merge into, whatever id it happens to carry.
        dst_exp = dst_by_name.get(name)
        if dst_exp is None:
            new_id = allocate_experiment_id(dst_store, preferred=src_exp.name)
            dst_exp = dst_store / new_id
            dst_exp.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_exp / "meta.yaml", dst_exp / "meta.yaml")
            update_meta(dst_exp / "meta.yaml", {
                "experiment_id": new_id,
                "artifact_location": paths.path_to_uri(dst_exp),
                "name": name,
            })
            dst_by_name[name] = dst_exp
            if verbose:
                print("    + experiment '%s' -> new local id %s" % (name, new_id))
        else:
            if verbose:
                print("    = experiment '%s' -> existing local id %s" % (name, dst_exp.name))

        # Carry over the experiment's non-run folders (tags, datasets, traces,
        # models) so nothing is silently dropped in the move.
        for extra in sorted(p for p in src_exp.iterdir()
                            if p.is_dir() and p.name in RESERVED_EXPERIMENT_DIRS):
            dst_extra = dst_exp / extra.name
            if not dst_extra.exists():
                shutil.copytree(extra, dst_extra)

        for src_run in run_dirs(src_exp):
            dst_run = dst_exp / src_run.name
            if dst_run.exists():
                if not overwrite:
                    skipped += 1
                    continue
                shutil.rmtree(dst_run)
            shutil.copytree(src_run, dst_run)
            # MLflow's FileStore refuses to resolve a run whose directory is missing
            # any of metrics/params/artifacts (_is_valid_run_directory), and a run
            # that logged none of a given kind arrives without that folder. Recreate
            # them so an otherwise fine imported run is not invisible in the UI.
            for required in ("metrics", "params", "artifacts"):
                (dst_run / required).mkdir(exist_ok=True)
            update_meta(dst_run / "meta.yaml", {
                "experiment_id": dst_exp.name,
                "artifact_uri": paths.path_to_uri(dst_run / "artifacts"),
            })
            # Provenance: which instance produced this run, and when it was imported.
            write_run_tag(dst_run, "source_instance", source_label)
            write_run_tag(dst_run, "imported_at", datetime.now().isoformat(timespec="seconds"))
            merged += 1

    return merged, skipped


def relink(store: Path) -> int:
    """Point every artifact_uri/artifact_location at the store's current location."""
    fixed = 0
    for exp in experiment_dirs(store):
        update_meta(exp / "meta.yaml", {"artifact_location": paths.path_to_uri(exp)})
        fixed += 1
        for run in run_dirs(exp):
            meta = run / "meta.yaml"
            if meta.is_file():
                update_meta(meta, {"artifact_uri": paths.path_to_uri(run / "artifacts")})
                fixed += 1
    return fixed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def default_tag(user_host: str, port: str) -> str:
    """Snapshot directory name: host, port and date, so repeat pulls stay distinct."""
    host = user_host.split("@")[-1].split(".")[0]
    return "%s-%s_%s" % (host, port, datetime.now().strftime("%Y%m%d-%H%M"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync vast.ai experiments onto the external disk and into one MLflow store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ssh-cmd", help="Vast.ai SSH string, e.g. 'ssh -p 12345 root@ssh5.vast.ai'")
    parser.add_argument("-p", "--port", help="SSH port (alternative to --ssh-cmd)")
    parser.add_argument("-H", "--host", help="user@host (alternative to --ssh-cmd)")
    parser.add_argument("--remote-root", default="/workspace",
                        help="Remote experiment root (default: /workspace)")
    parser.add_argument("--ssh-bin", default="ssh",
                        help="ssh executable to use (default: the one on PATH)")
    parser.add_argument("--include", nargs="+", default=DEFAULT_INCLUDES, metavar="GROUP",
                        help="What to pull: all, " + ", ".join(INCLUDE_GROUPS) +
                             " (default: " + " ".join(DEFAULT_INCLUDES) + ")")
    parser.add_argument("--dest", default=None,
                        help="Experiment root to sync into (default: resolved by src/config/paths.py)")
    parser.add_argument("--tag", default=None,
                        help="Snapshot name under <root>/imports (default: host-port_date)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Pull the snapshot but leave the local MLflow store alone")
    parser.add_argument("--merge-only", metavar="SNAPSHOT",
                        help="Skip the network; merge an already-pulled snapshot directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace runs that already exist in the local store")
    parser.add_argument("--relink", action="store_true",
                        help="Only repair artifact paths in the local store (after a drive-letter change)")
    parser.add_argument("--list", action="store_true",
                        help="Show the resolved layout and local store contents, then exit")
    args = parser.parse_args()

    if args.dest:
        os.environ[paths.ENV_VAR] = args.dest
    store = paths.mlruns_dir()

    print("=" * 68)
    print("   Experiment sync - vast.ai -> local archive")
    print("=" * 68)
    print(paths.describe())
    print()

    if args.list:
        experiments = experiment_dirs(store) if store.is_dir() else []
        if not experiments:
            print("No experiments in the local store yet.")
            print("Train locally, or pull one:  python sync_experiments.py --ssh-cmd \"...\"")
            return 0
        total = 0
        for exp in experiments:
            name = read_meta(exp / "meta.yaml").get("name") or exp.name
            runs = run_dirs(exp)
            total += len(runs)
            print("  [%s] %s: %d run(s)" % (exp.name, name, len(runs)))
        print("\n%d experiment(s), %d run(s) total." % (len(experiments), total))
        return 0

    if args.relink:
        count = relink(store)
        print("[ok] Repaired %d artifact path(s) in %s" % (count, store))
        return 0

    if args.merge_only:
        snapshot = Path(args.merge_only).expanduser().resolve()
        if not snapshot.exists():
            raise SystemExit("Snapshot not found: " + str(snapshot))
        source_label = snapshot.name
    else:
        port, user_host = args.port, args.host
        if args.ssh_cmd:
            port, user_host = parse_ssh_cmd(args.ssh_cmd)
        if not port or not user_host:
            entered = input("Paste the vast.ai SSH command (e.g. ssh -p 12345 root@ssh5.vast.ai): ").strip()
            if not entered:
                raise SystemExit("An SSH port and host are required.")
            port, user_host = parse_ssh_cmd(entered)
        if user_host and "@" not in user_host:
            user_host = "root@" + user_host

        source_label = args.tag or default_tag(user_host, port)
        snapshot = paths.imports_dir() / source_label
        pull(port, user_host, args.remote_root, args.include, snapshot, ssh_bin=args.ssh_bin)

    if args.no_merge:
        print("\nMerge skipped (--no-merge). Snapshot kept at " + str(snapshot))
        return 0

    src_store = find_mlruns(snapshot)
    if src_store is None:
        print("\n[Note] No mlruns tree inside %s - nothing to merge into the MLflow store." % snapshot)
        print("       (Add 'mlruns' to --include, or check --remote-root.)")
        return 0

    print("\n--> Merging %s into %s" % (src_store, store))
    merged, skipped = merge_store(src_store, store, source_label, overwrite=args.overwrite)
    print("[ok] Merged %d run(s), skipped %d already present%s"
          % (merged, skipped, " (use --overwrite to replace)" if skipped else ""))
    print("\nView them with:  powershell -File mlflow_local.ps1")
    # MLflow >= 3.16 needs the file-store opt-out; mlflow_local.ps1 sets it for you.
    print("     or by hand:  set MLFLOW_ALLOW_FILE_STORE=true")
    print("                  mlflow ui --backend-store-uri " + paths.path_to_uri(store))
    return 0


if __name__ == "__main__":
    sys.exit(main())
