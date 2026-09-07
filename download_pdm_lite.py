"""download_pdm_lite.py - pull as much of the PDM-Lite dataset as the disk will hold.

The setup script hard-coded WOR_DATASET_TOWNS=Town01 because the first rented instance
was small. That single town yields 9,640 frames across 129 routes, and a 15% split of it
leaves 17 held-out routes - which Challenge Group 13.19 identified as the reason the
architecture comparison could not resolve a 5% difference. Route count is the binding
constraint on what the evaluation can measure, and it is bounded by disk, not by choice.

So the town list is chosen at run time from the space actually available:

  * Real per-town sizes come from the HuggingFace HTTP API rather than a hard-coded
    table, so they cannot go stale when the dataset is revised. The plain HTTP endpoint
    is used instead of huggingface_hub's helpers because the size-carrying calls differ
    across hub versions (list_repo_tree, repo_info(files_metadata=...)) and the installed
    version varies by image.
  * Towns are taken smallest-first. Route diversity across maps matters more than raw
    frame count for a held-out set, and smallest-first maximises the number of distinct
    towns that fit in any given budget. Town12 and Town13 are ~117 GB each and are
    skipped unless explicitly asked for.
  * Each archive is downloaded, extracted, and deleted before the next is fetched, so
    peak usage is (what is already extracted) + (one archive) rather than the whole
    town twice over. On a disk sized for the final result that difference decides
    whether the run completes.

Already-extracted towns are skipped, so this is safe to re-run after an interruption or
after moving to a larger instance.

Usage:
  python download_pdm_lite.py --dry-run                    # show the plan, download nothing
  python download_pdm_lite.py                              # fill the disk, keeping 25 GB free
  python download_pdm_lite.py --reserve-gb 40              # leave more headroom
  python download_pdm_lite.py --towns Town01,Town03        # explicit, still disk-checked
  python download_pdm_lite.py --max-gb 60                  # cap regardless of free space
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from typing import Dict, List, Tuple

REPO_ID = "autonomousvision/PDM_Lite_Carla_LB2"
API_TREE = "https://huggingface.co/api/datasets/%s/tree/main/%%s?recursive=1" % REPO_ID
# Extracted size relative to the archive. PDM-Lite archives hold JPEGs and gzipped JSON,
# both already compressed, so extraction barely grows them - but never assume it shrinks.
EXTRACT_FACTOR = 1.05
# Town12/Town13 are ~117 GB each. Including them by default would let one town consume a
# whole disk and starve the map diversity that makes a held-out set worth having.
HUGE_TOWNS = {"Town12", "Town13"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default="/workspace/dataset/wor_trajectories")
    p.add_argument("--reserve-gb", type=float, default=25.0,
                   help="Free space to leave behind for checkpoints, CARLA and logs.")
    p.add_argument("--max-gb", type=float, default=None,
                   help="Hard cap on downloaded volume, regardless of free space.")
    p.add_argument("--towns", default=None,
                   help="Comma-separated explicit list. Still checked against the budget.")
    p.add_argument("--include-huge", action="store_true",
                   help="Allow Town12/Town13 (~117 GB each) into automatic selection.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    return p.parse_args()


def free_gb(path: str) -> float:
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe or "/").free / 1e9


def town_files(town: str) -> List[Tuple[str, int]]:
    """(path, size) for every file under `town`, via the version-independent HTTP API."""
    with urllib.request.urlopen(API_TREE % town, timeout=60) as r:
        entries = json.load(r)
    out = []
    for e in entries:
        if e.get("type") != "file":
            continue
        size = e.get("size") or (e.get("lfs") or {}).get("size") or 0
        out.append((e["path"], int(size)))
    return out


def discover(towns: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    listing = {}
    for t in towns:
        try:
            listing[t] = town_files(t)
        except Exception as exc:
            print("  [warn] could not list %s: %s" % (t, exc))
    return listing


def already_have(dest: str, town: str) -> bool:
    """A town counts as present only if it holds extracted route data.

    A directory containing nothing but leftover .zip files is a half-finished download,
    not a usable town, and must not be skipped - that is exactly the silent-failure shape
    this project keeps hitting (Group 2.7: training ran happily on unextracted archives).
    """
    d = os.path.join(dest, town)
    if not os.path.isdir(d):
        return False
    for root, _dirs, files in os.walk(d):
        if any(f.endswith(".json.gz") for f in files):
            return True
    return False


def plan(listing, dest, budget_gb, explicit):
    """Greedy smallest-first selection within the budget."""
    sizes = {t: sum(s for _p, s in fs) / 1e9 for t, fs in listing.items()}
    order = explicit if explicit else sorted(sizes, key=lambda t: sizes[t])

    chosen, used, skipped = [], 0.0, []
    for t in order:
        if t not in sizes:
            continue
        if already_have(dest, t):
            skipped.append((t, "already extracted"))
            continue
        need = sizes[t] * EXTRACT_FACTOR
        if used + need > budget_gb:
            skipped.append((t, "needs %.1f GB, only %.1f GB of budget left"
                            % (need, budget_gb - used)))
            continue
        chosen.append(t)
        used += need
    return chosen, used, skipped, sizes


def download_town(town: str, files: List[Tuple[str, int]], dest: str) -> int:
    """Fetch, extract and delete each archive in turn. Returns bytes extracted."""
    from huggingface_hub import hf_hub_download

    archives = [(p, s) for p, s in files if p.endswith(".zip")]
    # results.zip holds evaluation summaries, not driving logs - Group 9 established that
    # only the route archives carry measurements/ and rgb/.
    archives = [(p, s) for p, s in archives if not p.endswith("results.zip")]
    if not archives:
        print("    [warn] no route archives found for %s" % town)
        return 0

    total = 0
    for i, (path, size) in enumerate(archives, 1):
        started = time.time()
        print("    [%d/%d] %s (%.1f GB) ..." % (i, len(archives), path, size / 1e9),
              end="", flush=True)
        try:
            local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=path,
                                    local_dir=dest)
        except Exception as exc:
            print(" FAILED: %s" % exc)
            continue

        try:
            target = os.path.dirname(local)
            with zipfile.ZipFile(local) as z:
                z.extractall(target)
            extracted = size
            total += extracted
        except Exception as exc:
            print(" extract FAILED: %s" % exc)
            continue
        finally:
            try:
                os.remove(local)          # free the archive before fetching the next
            except OSError:
                pass

        mins, secs = divmod(int(time.time() - started), 60)
        print(" done in %dm%02ds | %.1f GB free" % (mins, secs, free_gb(dest)))
    return total


def count_routes(dest: str) -> Tuple[int, int]:
    routes, frames = set(), 0
    for root, _dirs, files in os.walk(dest):
        gz = [f for f in files if f.endswith(".json.gz")]
        if gz:
            routes.add(os.path.dirname(root))
            frames += len(gz)
    return len(routes), frames


def main():
    args = parse_args()
    dest = args.dest
    os.makedirs(dest, exist_ok=True)

    avail = free_gb(dest)
    budget = max(0.0, avail - args.reserve_gb)
    if args.max_gb is not None:
        budget = min(budget, args.max_gb)

    print("=" * 72)
    print("  PDM-Lite dataset download - sized to the disk")
    print("=" * 72)
    print("  destination : %s" % dest)
    print("  free space  : %.1f GB   (reserving %.1f GB)" % (avail, args.reserve_gb))
    print("  budget      : %.1f GB" % budget)

    explicit = [t.strip() for t in args.towns.split(",") if t.strip()] if args.towns else None
    candidates = explicit if explicit else None
    if candidates is None:
        try:
            with urllib.request.urlopen(
                    "https://huggingface.co/api/datasets/%s/tree/main" % REPO_ID, timeout=60) as r:
                top = json.load(r)
            candidates = [e["path"] for e in top
                          if e.get("type") == "directory" and e["path"].startswith("Town")]
        except Exception as exc:
            sys.exit("[FATAL] Could not list the dataset repository: %s" % exc)
        if not args.include_huge:
            candidates = [t for t in candidates if t not in HUGE_TOWNS]

    print("  candidates  : %s" % ", ".join(sorted(candidates)))
    print("\n--> Querying real sizes ...")
    listing = discover(candidates)
    if not listing:
        sys.exit("[FATAL] No town listings retrieved - check network access to huggingface.co.")

    chosen, used, skipped, sizes = plan(listing, dest, budget, explicit)

    print("\n  %-10s %10s   %s" % ("town", "size", "decision"))
    print("  " + "-" * 60)
    for t in sorted(sizes, key=lambda x: sizes[x]):
        why = "SELECTED" if t in chosen else \
              next((r for s, r in skipped if s == t), "not selected")
        print("  %-10s %8.1f GB   %s" % (t, sizes[t], why))

    print("\n  selected %d town(s), %.1f GB of a %.1f GB budget" % (len(chosen), used, budget))
    if not chosen:
        print("\n  Nothing to do. Either everything is present, or the budget is too small -")
        print("  lower --reserve-gb, or rent an instance with more disk.")
        return

    r0, f0 = count_routes(dest)
    if r0:
        print("  currently on disk: %d routes / %d frames" % (r0, f0))

    if args.dry_run:
        print("\n  --dry-run: nothing downloaded.")
        return

    for t in chosen:
        print("\n--> %s" % t)
        download_town(t, listing[t], dest)

    routes, frames = count_routes(dest)
    print("\n" + "=" * 72)
    print("  Done. %d routes / %d frames on disk (was %d / %d)" % (routes, frames, r0, f0))
    if r0:
        print("  A 15%% split now holds ~%d held-out routes, against ~%d before."
              % (round(routes * 0.15), round(r0 * 0.15)))
    print("  %.1f GB free remaining." % free_gb(dest))
    print("=" * 72)


if __name__ == "__main__":
    main()
