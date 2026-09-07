"""Checkpoint snapshotting and background writing for World on Rails training.

Split out of wor_trainer.py, which is the only caller. Two constraints shape this:

  - A frozen backbone is ~92% of the parameters and byte-for-byte identical in every
    epoch, so writing it each time (twice over, since "latest" and "best" both fire)
    costs far more wall time than the epoch itself. It ships once as
    frozen_backbone.pth and per-epoch files carry only the heads that change.
  - AdamW's exp_avg/exp_avg_sq buffers are updated in place on every subsequent step,
    so tensors must be snapshotted to CPU *before* a writer thread starts or
    serialization races the next epoch.
"""
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple
import os
import threading

import torch
import torch.nn as nn


def to_cpu(obj):
    """Recursively copies tensors in a (possibly nested) state dict to CPU."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


class CheckpointWriter:
    """Snapshots model weights and serializes them off the training critical path."""

    def __init__(self, model: nn.Module, save_dir: str, freeze_backbone: bool):
        self.model = model
        self.save_dir = save_dir
        self.frozen_keys: FrozenSet[str] = frozenset(
            k for k in model.state_dict() if k.startswith("encoder.")
        ) if freeze_backbone else frozenset()
        self._thread: Optional[threading.Thread] = None

    def state_dict_cpu(self, keys: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
        """Snapshots (a subset of) the model weights to CPU."""
        sd = self.model.state_dict()
        keys = sd.keys() if keys is None else keys
        return {k: sd[k].detach().to("cpu", copy=True) for k in keys}

    def trainable_state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        """Everything except the frozen backbone, which is written separately."""
        return self.state_dict_cpu(
            [k for k in self.model.state_dict() if k not in self.frozen_keys]
        )

    def write_frozen_backbone_once(self) -> None:
        """Writes the unchanging vision weights a single time, before training starts."""
        if not self.frozen_keys:
            return
        path = os.path.join(self.save_dir, "frozen_backbone.pth")
        torch.save({"model": self.state_dict_cpu(self.frozen_keys)}, path)
        n_total = len(self.model.state_dict())
        print(f"--> Wrote frozen backbone once ({len(self.frozen_keys)}/{n_total} tensors) to {path};"
              f" per-epoch checkpoints carry only the {n_total - len(self.frozen_keys)} trained head tensors.")

    def await_save(self) -> None:
        """Blocks until the previous epoch's write has finished."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def save_async(self, payloads: List[Tuple[Dict, str]]) -> None:
        """Writes checkpoints in a background thread.

        Only one write is ever in flight, so a slow disk throttles to one save per
        epoch rather than piling up threads.
        """
        self.await_save()

        def _write():
            for payload, path in payloads:
                tmp = path + ".tmp"
                torch.save(payload, tmp)
                os.replace(tmp, path)  # atomic: a killed run never leaves a half-file

        self._thread = threading.Thread(target=_write, daemon=False)
        self._thread.start()
