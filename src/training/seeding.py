"""Run-to-run reproducibility for training runs.

Nothing in the training path seeded anything before this module, which is fine for a
single run and fatal for a comparison. An ablation asks "did component X change the
result", and that question is unanswerable until the spread between two *identical*
runs is known and small relative to the effect being claimed. Seeding is what makes
that spread measurable; `--seed` sweeps are what measure it.

Two levels are offered because they cost differently:

  - `seed_everything(seed)` fixes weight initialisation, dropout, and data order. This
    is where nearly all run-to-run variance lives, and it is free.
  - `deterministic=True` additionally pins cuDNN kernel selection and refuses
    nondeterministic CUDA kernels. It gives bitwise reproducibility and costs real
    throughput, because it disables the cuDNN autotuner that `--policy_arch cnn`
    benefits from. Use it to reproduce a specific run, not to run a sweep.
"""
from typing import Callable, Optional
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seeds Python, NumPy and Torch (CPU and all CUDA devices). Returns the seed.

    `PYTHONHASHSEED` is set here for completeness but only takes effect for processes
    started afterwards - Python reads it at interpreter startup, so set it in the
    environment if hash-order reproducibility across processes matters.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS needs this set before its first call to behave deterministically on
        # CUDA >= 10.2; setting it here covers the common case of seeding at startup.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # torch < 1.11 has no warn_only
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as exc:
                print(f"[Warning] Could not enable deterministic algorithms: {exc}")
        except Exception as exc:
            print(f"[Warning] Could not enable deterministic algorithms: {exc}")
        print(f"--> Seeded with {seed} (DETERMINISTIC: cuDNN autotuner disabled, throughput will drop).")
    else:
        print(f"--> Seeded with {seed} (init + data order fixed; cuDNN autotuner still active).")

    return seed


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Per-worker seeding for DataLoader subprocesses.

    Torch reseeds its own RNG per worker automatically, but NumPy's is inherited by
    fork/spawn unchanged - so every worker draws the *same* NumPy random numbers. Any
    NumPy-based augmentation or synthetic sampling is then silently duplicated across
    workers, which both correlates the batch and makes the run unreproducible when
    `--num_workers` changes.
    """
    def _init(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _init


def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Generator for DataLoader shuffling, so sample order is a function of the seed."""
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
