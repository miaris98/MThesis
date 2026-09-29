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
import math
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


def load_trainable_state(model: nn.Module, state: Dict[str, torch.Tensor], frozen_keys: Iterable[str],
                         source: str) -> None:
    """Loads a resume checkpoint's head weights, and refuses a load that left any of them untouched.

    strict=False is needed because per-epoch files omit the frozen backbone, but on its own it also accepted a
    checkpoint whose keys carry torch.compile's "_orig_mod." prefix into an uncompiled model (or the reverse):
    nothing matched, the heads kept their random init, and the run continued as a "resume" with the trained
    optimizer state. Keys are re-prefixed to this model's convention first, and a trainable tensor still
    missing afterwards is an error.
    """
    compiled = any(k.startswith("_orig_mod.") for k in model.state_dict())

    def _key(k: str) -> str:
        k = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        return "_orig_mod." + k if compiled else k

    missing, _ = model.load_state_dict({_key(k): v for k, v in state.items()}, strict=False)
    frozen = set(frozen_keys)
    missing = [k for k in missing if k not in frozen]
    if missing:
        raise RuntimeError(
            f"{source} does not hold {len(missing)} of this model's trainable tensors "
            f"(e.g. {', '.join(missing[:4])}). It was written by a different configuration; resuming "
            f"would train randomly initialised heads with that run's optimizer state.")


def best_metric_so_far(save_dir: str, select_on: str, resume_ckpt: Dict) -> float:
    """The best `select_on` value a resumed run has already promoted to best_model.pth.

    The resumed epoch's own metric is only an upper bound: resuming a 20-epoch screen whose best epoch was 15
    used to set the threshold to epoch 20's value, so a later epoch worse than 15 overwrote best_model.pth.
    The value stored in save_dir's best_model.pth is the real threshold; the resumed epoch's metric still
    counts in case best_model.pth was not copied along with the snapshot.
    """
    values = [resume_ckpt.get("metrics", {}).get(select_on)]
    best_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_path):
        try:
            values.append(torch.load(best_path, map_location="cpu").get("metrics", {}).get(select_on))
        except Exception as exc:  # a corrupt best file must not stop the resume, only be reported
            print(f"[Warning] could not read {best_path} ({exc}); best selection restarts from the "
                  f"resumed epoch's {select_on}.")
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return min(finite) if finite else float("inf")


class CheckpointWriter:
    """Snapshots model weights and serializes them off the training critical path."""

    def __init__(self, model: nn.Module, save_dir: str, freeze_backbone: bool):
        self.model = model
        self.save_dir = save_dir
        # torch.compile wraps state_dict keys as "_orig_mod.encoder...." rather than
        # "encoder....". Missing that prefix here silently made frozen_keys empty for every
        # compiled run: no exception, no frozen_backbone.pth, and every per-epoch checkpoint
        # quietly grew from heads-only to the full model (encoder included) instead - found
        # 2026-09-14 comparing checkpoint file sizes against an uncompiled run's.
        self.frozen_keys: FrozenSet[str] = frozenset(
            k for k in model.state_dict()
            if k.startswith("encoder.") or k.startswith("_orig_mod.encoder.")
        ) if freeze_backbone else frozenset()
        self._thread: Optional[threading.Thread] = None
        # An exception in the writer thread (disk full, the mount gone) used to end that thread with a
        # traceback on stderr while training carried on for the rest of the run writing no checkpoints.
        # It is kept here and raised in the training thread at the next save or await.
        self._error: Optional[BaseException] = None

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
        tmp = path + ".tmp"
        torch.save({"model": self.state_dict_cpu(self.frozen_keys)}, tmp)
        os.replace(tmp, path)  # atomic, like the per-epoch saves: a live sync never copies a half-written backbone
        n_total = len(self.model.state_dict())
        print(f"--> Wrote frozen backbone once ({len(self.frozen_keys)}/{n_total} tensors) to {path};"
              f" per-epoch checkpoints carry only the {n_total - len(self.frozen_keys)} trained head tensors.")

    def await_save(self) -> None:
        """Blocks until the previous epoch's write has finished; raises if that write failed."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            err, self._error = self._error, None
            raise RuntimeError(f"checkpoint write failed: {err!r}") from err

    def save_async(self, payloads: List[Tuple[Dict, str]]) -> None:
        """Writes checkpoints in a background thread.

        Only one write is ever in flight, so a slow disk throttles to one save per
        epoch rather than piling up threads.
        """
        self.await_save()

        def _write():
            for payload, path in payloads:
                tmp = path + ".tmp"
                try:
                    torch.save(payload, tmp)
                    os.replace(tmp, path)  # atomic: a killed run never leaves a half-file
                except BaseException as exc:
                    self._error = exc
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    return

        self._thread = threading.Thread(target=_write, daemon=False)
        self._thread.start()
