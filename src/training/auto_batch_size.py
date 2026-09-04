"""VRAM-aware batch size auto-tuner.

Probes the largest batch size whose peak VRAM usage - forward + backward +
optimizer step, not just a forward pass - stays under (total VRAM - headroom),
then discards the probe model/optimizer so real training starts from an
unperturbed model. The headroom exists because this GPU may not be
exclusively ours: another process (e.g. an online PPO/SAC trainer sharing the
same box) needs room to allocate too, and CUDA's allocator/fragmentation
overhead means "fits once" isn't the same as "fits with margin for real
training runs, which vary batch-to-batch."
"""
from typing import Callable, Dict, Optional
import gc
import subprocess
import torch
import torch.nn as nn

try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    from torch.amp import GradScaler, autocast


def print_gpu_process_usage(device_index: int = 0) -> None:
    """Prints every process currently holding VRAM on this GPU, via nvidia-smi -
    including zombie CUDA contexts (dead process, driver never released the
    memory) that torch.cuda's own allocator stats can't see, since those only
    describe THIS process's allocations. Best-effort: silently no-ops if
    nvidia-smi isn't available.

    Does NOT claim to know whether a listed PID is alive or dead: nvidia-smi
    talks to the driver at the host level and reports host-namespace PIDs,
    which on a containerized GPU (vast.ai and similar) generally do not exist
    in this container's own PID namespace at all - `kill -0 <pid>` fails with
    "No such process" for those regardless of whether the process is actually
    alive and well inside its own container. Treating that as proof of death
    is exactly how a live, merely-Ctrl+Z-suspended training run got
    misdiagnosed as an unrecoverable leak more than once. If you started the
    process yourself, `jobs` + `kill -9 %N` (shell job control, not the PID
    shown here) is the first thing to try - it targets the container-local
    PID your shell actually owns.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits", "-i", str(device_index)],
            capture_output=True, text=True, timeout=10
        )
        rows = [r.strip() for r in out.stdout.strip().splitlines() if r.strip()]
        if not rows:
            print("[auto_batch_size] No other processes currently hold VRAM on this GPU.")
            return
        print(f"[auto_batch_size] {len(rows)} process(es) already holding VRAM on GPU {device_index} "
              f"(PIDs are nvidia-smi's host-level view - if one of these is your own suspended run, "
              f"use `jobs` + `kill -9 %N` instead of `kill -9 <pid>`, which will likely say "
              f"'No such process' here regardless of whether it's actually alive):")
        for row in rows:
            pid, used_mb = [p.strip() for p in row.split(",")]
            print(f"    PID {pid}: {used_mb}MB")
    except Exception:
        pass


def find_max_batch_size(
    model_factory: Callable[[], nn.Module],
    optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
    batch_factory: Callable[[int], Dict[str, torch.Tensor]],
    loss_fn: Callable[[nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
    device: str = "cuda",
    start_batch: int = 8,
    max_batch: int = 512,
    headroom_mb: float = 2048.0,
    use_amp: bool = True
) -> int:
    """Binary-searches the largest batch size that fits under the VRAM budget.

    Each candidate gets a fresh model + optimizer (from the factories) so the
    probe never touches the weights real training will start from, and runs
    one genuine forward + backward + optimizer.step() so lazily-allocated
    optimizer state (e.g. AdamW's exp_avg/exp_avg_sq buffers, only created on
    the first .step()) is included in the measurement rather than
    underestimated by a forward-only or backward-only probe.
    """
    if device != "cuda" or not torch.cuda.is_available():
        print(f"[auto_batch_size] Not on CUDA - using start_batch={start_batch}.")
        return start_batch

    # mem_get_info() reports the DEVICE-WIDE free/total split (every process, this one
    # included), unlike torch.cuda.max_memory_allocated() which only ever sees this
    # process's own allocations. Budgeting off total_memory instead of actual free
    # memory is exactly what caused an OOM on a GPU that had other processes (including
    # zombie CUDA contexts - dead process, but the driver never released their VRAM)
    # already holding several GB before this process allocated anything.
    free_mb, total_mb = (x / (1024 ** 2) for x in torch.cuda.mem_get_info())
    print_gpu_process_usage()
    budget_mb = free_mb - headroom_mb
    if budget_mb <= 0:
        raise ValueError(
            f"headroom_mb ({headroom_mb:.0f}) exceeds currently free VRAM ({free_mb:.0f}MB of {total_mb:.0f}MB total). "
            f"Other processes are holding the rest - see the usage listed above."
        )
    print(f"[auto_batch_size] Total VRAM: {total_mb:.0f}MB | Free right now: {free_mb:.0f}MB | "
          f"Headroom: {headroom_mb:.0f}MB | Budget: {budget_mb:.0f}MB")

    def _probe(bs: int) -> Optional[float]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = model_factory().to(device)
        optimizer = optimizer_factory(model)
        scaler = GradScaler(enabled=use_amp)
        try:
            batch = batch_factory(bs)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                loss = loss_fn(model, batch)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            # reserved, not allocated: PyTorch's caching allocator reserves more than it
            # allocates, and it's the reserved amount that actually occupies device
            # memory other processes can't use.
            return torch.cuda.max_memory_reserved() / (1024 ** 2)
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()

    def _log(bs: int, peak: Optional[float], fits: bool) -> None:
        status = f"{peak:.0f}MB" if peak is not None else "OOM"
        print(f"[auto_batch_size] batch_size={bs:4d} -> peak={status}{' (fits)' if fits else ' (too high)'}")

    # 1. Exponential search for an upper bound that doesn't fit.
    bs, last_good, hi = start_batch, None, None
    while bs <= max_batch:
        peak = _probe(bs)
        fits = peak is not None and peak <= budget_mb
        _log(bs, peak, fits)
        if fits:
            last_good = bs
            bs *= 2
        else:
            hi = bs
            break

    if last_good is None:
        raise RuntimeError(
            f"Even the starting batch size {start_batch} does not fit the "
            f"{budget_mb:.0f}MB VRAM budget (headroom={headroom_mb:.0f}MB). "
            f"Try a smaller start_batch or a smaller backbone."
        )
    if hi is None:
        print(f"[auto_batch_size] Reached max_batch={max_batch} while still fitting - capping there.")
        return last_good

    # 2. Binary search between last_good (fits) and hi (doesn't).
    lo = last_good
    while hi - lo > 1:
        mid = (lo + hi) // 2
        peak = _probe(mid)
        fits = peak is not None and peak <= budget_mb
        _log(mid, peak, fits)
        if fits:
            lo = mid
            last_good = mid
        else:
            hi = mid

    print(f"[auto_batch_size] Selected batch_size={last_good} (peak fits under {budget_mb:.0f}MB budget).")
    return last_good
