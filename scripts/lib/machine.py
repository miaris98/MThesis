#!/usr/bin/env python3
"""What this box actually has, and how big each job should be on it.

Every launcher used to hard-code its sizes (12 Ray CPUs, 4 OMP threads, `taskset -c 16-31`, 5 lanes),
which was right for one box and wrong for the next. Three recurring traps are measured here instead of
assumed:

* `os.cpu_count()` reports every host CPU, not the container's quota. Box W showed 48 while its cgroup
  allowed 11.5, and 48 Ray workers exhausted `pids.max` (S-086).
* `pids.max` caps processes plus threads, so thread-heavy jobs (Ray, CARLA, torch) must be sized
  against it, not only against CPUs.
* CUDA numbers GPUs fastest-first by default while CARLA's `-graphicsadapter` and nvidia-smi use PCI
  order, so the same index can mean two different cards (see `common.sh`, which exports
  CUDA_DEVICE_ORDER=PCI_BUS_ID).

Standard library only: it runs under the system python3 on boxes where torch is not installed.

    python scripts/lib/machine.py              # JSON summary
    python scripts/lib/machine.py --export     # shell `export` lines, for `eval "$(...)"` in launchers
    python scripts/lib/machine.py --field ezv2_num_cpus
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

#: CARLA eval lane footprint (S-093/S-098): ~5 cores and one CARLA server (~5.7 GB VRAM, S-012) plus
#: the agent (~1.5 GB) per lane.
CARLA_CORES_PER_LANE = 5.0
CARLA_VRAM_GB_PER_LANE = 7.5
#: Keep this share of VRAM free (CLAUDE.md: ~10-15 % headroom against CUDA OOM).
VRAM_HEADROOM = 0.15
#: EZ-V2 with Ray used ~2,450 pids at 12 CPUs (S-086): ~200 pids per Ray CPU, plus margin.
PIDS_PER_RAY_CPU = 220


@dataclass
class Gpu:
    index: int
    name: str
    memory_total_gb: float
    memory_used_gb: float
    compute_capability: Optional[float] = None

    @property
    def memory_free_gb(self) -> float:
        return max(0.0, self.memory_total_gb - self.memory_used_gb)

    @property
    def bf16(self) -> Optional[bool]:
        """Native bf16 needs Ampere (8.0) or newer; Turing emulates it slowly (2588bd1)."""
        return None if self.compute_capability is None else self.compute_capability >= 8.0


@dataclass
class Machine:
    cpus: float
    host_cpus: int
    pids_max: Optional[int]
    memory_gb: Optional[float]
    gpus: List[Gpu] = field(default_factory=list)


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cpu_quota(root: Path = Path("/sys/fs/cgroup")) -> Optional[float]:
    """CPUs the cgroup allows (cgroup v2 `cpu.max`, else v1 CFS quota); None if unlimited/unknown."""
    v2 = _read(root / "cpu.max")
    if v2:
        quota, _, period = v2.partition(" ")
        if quota != "max" and period:
            return int(quota) / int(period)
        return None
    quota = _read(root / "cpu" / "cpu.cfs_quota_us") or _read(root / "cpu,cpuacct" / "cpu.cfs_quota_us")
    period = _read(root / "cpu" / "cpu.cfs_period_us") or _read(root / "cpu,cpuacct" / "cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period)
    return None


def pids_max(root: Path = Path("/sys/fs/cgroup")) -> Optional[int]:
    """Process+thread cap of the cgroup; None if unlimited/unknown."""
    for p in (root / "pids.max", root / "pids" / "pids.max"):
        v = _read(p)
        if v and v != "max":
            return int(v)
    return None


def memory_limit_gb(root: Path = Path("/sys/fs/cgroup"), meminfo: Path = Path("/proc/meminfo")) -> Optional[float]:
    """The smaller of the cgroup memory limit and physical RAM, in GB."""
    limits = []
    for p in (root / "memory.max", root / "memory" / "memory.limit_in_bytes"):
        v = _read(p)
        if v and v != "max" and int(v) < 1 << 60:  # v1 reports ~2^63 for "unlimited"
            limits.append(int(v) / 1e9)
    info = _read(meminfo)
    if info:
        for line in info.splitlines():
            if line.startswith("MemTotal:"):
                limits.append(int(line.split()[1]) * 1024 / 1e9)
    return min(limits) if limits else None


def usable_cpus(root: Path = Path("/sys/fs/cgroup")) -> float:
    """min(cgroup quota, CPUs this process may be scheduled on, host CPUs)."""
    candidates = [float(os.cpu_count() or 1)]
    if hasattr(os, "sched_getaffinity"):
        candidates.append(float(len(os.sched_getaffinity(0))))  # taskset / cpuset
    quota = cpu_quota(root)
    if quota:
        candidates.append(quota)
    return min(candidates)


def query_gpus(nvidia_smi: Optional[str] = None) -> List[Gpu]:
    """GPUs from nvidia-smi in PCI order; [] without a driver. compute_cap needs driver >= 510."""
    exe = nvidia_smi or shutil.which("nvidia-smi")
    if not exe:
        return []
    base = "index,name,memory.total,memory.used"
    for fields in (base + ",compute_cap", base):
        try:
            out = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return []
        if out.returncode == 0:
            return parse_nvidia_smi(out.stdout, with_cc=fields.endswith("compute_cap"))
    return []


def parse_nvidia_smi(text: str, with_cc: bool) -> List[Gpu]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        cc = None
        if with_cc and len(parts) > 4:
            try:
                cc = float(parts[4])
            except ValueError:
                cc = None
        gpus.append(Gpu(int(parts[0]), parts[1], float(parts[2]) / 1024, float(parts[3]) / 1024, cc))
    return gpus


def detect(root: Path = Path("/sys/fs/cgroup")) -> Machine:
    return Machine(cpus=usable_cpus(root), host_cpus=os.cpu_count() or 1, pids_max=pids_max(root),
                   memory_gb=memory_limit_gb(root), gpus=query_gpus())


def recommend(m: Machine) -> dict:
    """Job sizes for this box. Each value follows the rule in CLAUDE.md or a measured incident."""
    cpus = max(1.0, m.cpus)
    rec = {
        # DataLoader: leave 2 cores for the main process and the GPU feeder; 4..16 workers.
        "dataloader_workers": int(min(16, max(4 if cpus >= 6 else 1, math.floor(cpus) - 2))),
        # Ray for EZ-V2: the CPU quota, capped by what pids.max can hold (S-086: 11.5 CPUs -> 12).
        "ezv2_num_cpus": int(max(1, math.ceil(cpus))),
        # Torch intra-op threads per CARLA agent next to its simulator (queue_A9c.sh used 4).
        "eval_omp_threads": int(max(1, min(4, math.floor(cpus / 4)))),
    }
    if m.pids_max:
        rec["ezv2_num_cpus"] = int(max(1, min(rec["ezv2_num_cpus"], m.pids_max // PIDS_PER_RAY_CPU)))
    lanes_by_cpu = int(cpus // CARLA_CORES_PER_LANE)
    lanes_by_vram = sum(int(g.memory_free_gb * (1 - VRAM_HEADROOM) // CARLA_VRAM_GB_PER_LANE) for g in m.gpus)
    rec["carla_lanes"] = int(max(0, min(lanes_by_cpu, lanes_by_vram) if m.gpus else 0))
    rec["carla_lanes_per_gpu"] = {g.index: int(g.memory_free_gb * (1 - VRAM_HEADROOM) // CARLA_VRAM_GB_PER_LANE)
                                  for g in m.gpus}
    rec["bf16"] = bool(m.gpus) and all(g.bf16 for g in m.gpus)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true", help="print shell export lines")
    ap.add_argument("--field", help="print one recommendation or machine field")
    args = ap.parse_args()
    m = detect()
    rec = recommend(m)
    if args.field:
        value = rec.get(args.field, getattr(m, args.field, None))
        if value is None:
            ap.error(f"unknown field {args.field}")
        print(str(value).lower() if isinstance(value, bool) else value)
        return 0
    if args.export:
        print(f"export MACHINE_CPUS={m.cpus:g} MACHINE_GPUS={len(m.gpus)} MACHINE_PIDS_MAX={m.pids_max or 0}")
        for k, v in rec.items():
            if not isinstance(v, dict):
                print(f"export REC_{k.upper()}={str(v).lower() if isinstance(v, bool) else v}")
        return 0
    print(json.dumps({"machine": {**asdict(m), "gpus": [{**asdict(g), "bf16": g.bf16} for g in m.gpus]},
                      "recommend": rec}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
