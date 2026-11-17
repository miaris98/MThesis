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
                         source: str) -> List[str]:
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

    own = model.state_dict()
    incoming = {_key(k): v for k, v in state.items()}
    # TODO A54: the vision input of a space-to-depth model (vision_proj, the vision position table) has another shape than the resumed checkpoint's: those two tensors
    # start from their own init and the rest of the trunk (the driving knowledge) is kept. Any OTHER shape mismatch stays an error.
    reshaped = [k for k, v in incoming.items() if k in own and own[k].shape != v.shape]
    ok_reshape = [k for k in reshaped if k.startswith(("vision_proj.", "trunk.vision_pos"))]
    # 2026-11-07 (WOR_RESUME_WIDEN=1, set by train_wor.py for the prev-frame / ray-geometry / route-window arms): a vision input that only
    # GREW keeps the checkpoint's weights in its leading slice. vision_proj.weight gets zeros in the new input columns (the added channels
    # start with no effect, so the arm starts exactly at the checkpoint); trunk.vision_pos keeps the old tokens' rows, new token rows keep
    # their own init. The s2d arms (a different channel layout) keep the old behaviour: no WOR_RESUME_WIDEN, own init.
    # 2026-11-07, full-map arms (WOR_PCA_BASIS set by train_wor.py --pca_full): the checkpoint's vision_proj read 1512-channel pooled cells; the
    # new one reads K PCA coordinates of one cell. x ~ mean + basis^T z, so W_new = W_old basis^T and b_new = b_old + W_old mean give each cell
    # token what the old projection would give that cell's features. trunk.vision_pos 4x4 -> HxW: every fine cell starts at its 4x4 parent's row.
    special = []
    pca_basis = os.environ.get("WOR_PCA_BASIS")
    if pca_basis and "vision_proj.weight" in [k.split("_orig_mod.")[-1] for k in ok_reshape]:
        st = torch.load(pca_basis, map_location="cpu")
        wk = [k for k in ok_reshape if k.endswith("vision_proj.weight")][0]
        bk = wk[:-len("weight")] + "bias"
        w_old = incoming[wk].double()
        mean, basis = st["mean"].double().to(w_old.device), st["basis"].double().to(w_old.device)   # (C,), (K, C), on the checkpoint's device
        if w_old.shape[1] == basis.shape[1] and own[wk].shape[1] == basis.shape[0]:
            incoming[wk] = (w_old @ basis.T).to(own[wk].dtype)
            incoming[bk] = (incoming[bk].double() + w_old @ mean).to(own[bk].dtype)
            special.append(wk)
    grid_hw = os.environ.get("WOR_VISION_GRID_HW", "")
    for k in [k for k in ok_reshape if k.endswith("trunk.vision_pos")]:
        old, new = incoming[k], own[k]
        if grid_hw and "x" in grid_hw and old.shape[1] == 16:
            gh, gw = (int(v) for v in grid_hw.lower().split("x"))
            if new.shape[1] == gh * gw:
                rows = torch.arange(gh).repeat_interleave(gw) * 4 // gh
                cols = torch.arange(gw).repeat(gh) * 4 // gw
                incoming[k] = old[:, (rows * 4 + cols).to(old.device)].to(new.dtype).clone()
                special.append(k)
    if special:
        print(f"--> {len(special)} vision-input tensors mapped from the checkpoint for the full map: {', '.join(special)}", flush=True)
        ok_reshape = [k for k in ok_reshape if k not in special]
    if ok_reshape and os.environ.get("WOR_RESUME_WIDEN", "0") == "1":
        widened = []
        for k in ok_reshape:
            old, new = incoming[k], own[k].detach().clone()
            if old.dim() == new.dim() and all(o <= n for o, n in zip(old.shape, new.shape)):
                if k.endswith("vision_proj.weight"):
                    new[:, old.shape[1]:] = 0.0
                new[tuple(slice(0, s) for s in old.shape)] = old.to(new.dtype)
                incoming[k] = new
                widened.append(k)
        if widened:
            print(f"--> {len(widened)} vision-input tensors widened from the checkpoint (old weights kept, new inputs start at zero / own init): "
                  f"{', '.join(widened)}", flush=True)
        ok_reshape = [k for k in ok_reshape if k not in widened]
        changed_widened = widened
    else:
        changed_widened = []
    changed_widened = changed_widened + special
    if ok_reshape:
        print(f"--> {len(ok_reshape)} vision-input tensors changed shape and start from their own init: {', '.join(ok_reshape)}", flush=True)
        incoming = {k: v for k, v in incoming.items() if k not in ok_reshape}
    missing, unexpected = model.load_state_dict(incoming, strict=False)
    # (review finding) a checkpoint's trained expert weights that this model does not build would be dropped without a word
    _dropped = [k for k in unexpected if k.startswith(("vis_moe.", "trunk.moe.", "speed_moe."))]
    if _dropped:
        raise RuntimeError(f"{source} holds {len(_dropped)} expert tensors this model does not build (e.g. {', '.join(_dropped[:3])}); "
                           f"resuming would discard them. Build the model with the checkpoint's expert counts.")
    missing = [k for k in missing if k not in ok_reshape]
    frozen = set(frozen_keys)
    missing = [k for k in missing if k not in frozen]
    new_heads = [k for k in missing if k.startswith(("path_head.", "path_embed.", "q_head.", "v_head.", "q_sa_head.", "trunk.moe.", "speed_moe.", "vis_moe.", "cell_head.", "attn_pool.", "pool_queries", "cell_pos",
                                                       "win_head.", "win_pre_head.", "win_feed.", "adapter.", "full_head.", "qd_queries", "qd_cell_proj.", "qd.", "qd_out.",
                                                       "reason_head.", "reason_embed."))]   # 2026-11-09 query decoder; TODO A15 / A71 / A84, 2026-11-07 MoE: heads a resumed arm adds on purpose
    if new_heads:
        print(f"--> {len(new_heads)} new-head tensors start from their own init (zero output): {', '.join(new_heads[:3])} ...", flush=True)
    missing = [k for k in missing if k not in new_heads]
    changed = new_heads + ok_reshape + changed_widened    # tensors whose shape or init differs: the resumed optimizer state no longer fits them
    if missing:
        raise RuntimeError(
            f"{source} does not hold {len(missing)} of this model's trainable tensors "
            f"(e.g. {', '.join(missing[:4])}). It was written by a different configuration; resuming "
            f"would train randomly initialised heads with that run's optimizer state.")
    return changed


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
