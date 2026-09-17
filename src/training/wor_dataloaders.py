"""DataLoader construction and the route-group train/validation split.

Split out of `wor_dataset.py` when that module outgrew this repo's 500-line limit. The seam is
between "what a sample is" (the Dataset, still there) and "how samples are batched, seeded and
divided into train/val" (here). The split logic in particular is worth isolating: it divides on
*route* boundaries rather than frames, which is the difference between a held-out set that
measures generalisation and one that measures memorisation of neighbouring frames.

Every name here is re-exported from `wor_dataset`, so existing imports keep working.
"""
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
from torch.utils.data import DataLoader, Subset

from src.training.seeding import make_generator, make_worker_init_fn
from src.training.wor_dataset import WorldOnRailsDataset


def _wrap_loader(dataset, batch_size: int, num_workers: int, is_train: bool,
                 seed: Optional[int] = None) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=num_workers > 0
    )
    # Sample order and per-worker RNG are seeded so two runs of the same config differ
    # only by what is being ablated. See src/training/seeding.py for why NumPy needs
    # explicit per-worker seeding where Torch does not.
    if seed is not None:
        if is_train:
            kwargs["generator"] = make_generator(seed)
        if num_workers > 0:
            kwargs["worker_init_fn"] = make_worker_init_fn(seed)
    # Only pass prefetch_factor in the multiprocessing case. Torch accepted an
    # explicit None here from 2.0 onward, but older versions reject it outright
    # ("prefetch_factor option could only be specified in multiprocessing"), which
    # made num_workers=0 - the default on Windows, and what the tests use - fail
    # before a single batch was read.
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def create_wor_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    is_train: bool = True,
    synthetic_samples: int = 0,
    cache_decoded: bool = True,
    route_points: int = 4,
    img_size: Tuple[int, int] = (256, 256),
    crop_bottom_frac: float = 0.0,
    route_overlay: bool = False,
    overlay_kwargs: Optional[Dict] = None,
    use_augmented_camera: bool = False
) -> DataLoader:
    """Creates a DataLoader for World on Rails training/validation."""
    dataset = WorldOnRailsDataset(
        data_dir=data_dir,
        is_train=is_train,
        synthetic_samples=synthetic_samples,
        cache_decoded=cache_decoded,
        route_points=route_points,
        img_size=img_size,
        crop_bottom_frac=crop_bottom_frac,
        route_overlay=route_overlay,
        overlay_kwargs=overlay_kwargs,
        use_augmented_camera=use_augmented_camera
    )
    return _wrap_loader(dataset, batch_size, num_workers, is_train)



def route_group_split(
    dataset, val_split: float, split_seed: int, fold: int = 0, num_folds: int = 1
) -> Tuple[List[int], List[int], Dict[str, List[int]], List[str]]:
    """Partitions `dataset` into train/val indices on route boundaries.

    Extracted so that anything scoring a trained checkpoint can reconstruct the exact
    partition that checkpoint was trained under. A second, separately-maintained copy
    of this logic would be worse than none: an evaluation that re-derives the split
    slightly differently silently grades the model on frames it was trained on, and
    reports an optimistic number with no error to indicate it.

    Returns `(train_idx, val_idx, groups, shuffled_keys)`. `groups` maps each route key
    to its frame indices, which is what lets a caller resample at route granularity
    rather than frame granularity.
    """
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(dataset.samples):
        key = ""
        if isinstance(s, dict):
            key = os.path.dirname(os.path.dirname(s.get("rgb_path", ""))) or s.get("route_dir", "")
        groups.setdefault(key or str(i), []).append(i)

    rng = np.random.RandomState(split_seed)
    keys = sorted(groups)
    rng.shuffle(keys)

    if num_folds and num_folds > 1:
        # K-fold over routes: the shuffled route list is cut into num_folds contiguous
        # blocks and block `fold` is held out. Across all folds every route is held out
        # exactly once, so a model scored fold-by-fold is scored on the whole dataset
        # rather than on one 15% draw. That is the entire point: sampling error falls
        # with the square root of the number of independent held-out routes, and with
        # 129 routes against 17 that is a factor of 2.8 on every interval.
        #
        # Contiguous blocks of the *shuffled* list, not a modulo stride, so the folds
        # stay disjoint and reproducible from (split_seed, num_folds) alone.
        fold = int(fold) % int(num_folds)
        bounds = [round(len(keys) * f / num_folds) for f in range(num_folds + 1)]
        val_keys = keys[bounds[fold]:bounds[fold + 1]]
        val_idx = [i for k in val_keys for i in groups[k]]
    else:
        target = int(round(len(dataset) * val_split))
        val_idx = []
        for k in keys:
            if len(val_idx) >= target:
                break
            val_idx.extend(groups[k])

    val_set = set(val_idx)
    train_idx = [i for i in range(len(dataset)) if i not in val_set]
    return train_idx, val_idx, groups, keys


def create_wor_train_val_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    synthetic_samples: int = 0,
    cache_decoded: bool = True,
    route_points: int = 4,
    val_data_dir: Optional[str] = None,
    val_split: float = 0.0,
    split_seed: int = 0,
    seed: Optional[int] = None,
    fold: int = 0,
    num_folds: int = 1,
    img_size: Tuple[int, int] = (256, 256),
    crop_bottom_frac: float = 0.0,
    route_overlay: bool = False,
    overlay_kwargs: Optional[Dict] = None,
    feature_cache_tag: Optional[str] = None,
    use_augmented_camera: bool = False
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Creates the training loader and, when asked for, a held-out validation loader.

    A separate `val_data_dir` is preferred when one exists, because a random split of
    `data_dir` divides *frames*, not routes - consecutive frames of the same route are
    near-duplicates, so a frame-level split leaks and reports an optimistic number.
    The split is offered anyway because no validation at all is strictly worse: the
    trainer previously selected its "best" checkpoint on training loss, which cannot
    distinguish a model that generalises from one that has memorised.

    Routes are kept intact where the layout exposes them (PDM-Lite samples carry their
    source `rgb_path`, whose parent directory identifies the route), so the split falls
    on route boundaries rather than frame boundaries whenever that information exists.
    """
    train_ds = WorldOnRailsDataset(
        data_dir=data_dir, is_train=True, synthetic_samples=synthetic_samples,
        cache_decoded=cache_decoded, route_points=route_points,
        img_size=img_size, crop_bottom_frac=crop_bottom_frac,
        route_overlay=route_overlay, overlay_kwargs=overlay_kwargs,
        feature_cache_tag=feature_cache_tag, use_augmented_camera=use_augmented_camera
    )

    if val_data_dir:
        # use_augmented_camera is passed through but is a no-op here: WorldOnRailsDataset
        # gates it on is_train, and validation should keep measuring on-route driving so the
        # held-out metric stays comparable to runs that don't use this flag (see S-020).
        val_ds = WorldOnRailsDataset(
            data_dir=val_data_dir, is_train=False, synthetic_samples=0,
            cache_decoded=cache_decoded, route_points=route_points,
            img_size=img_size, crop_bottom_frac=crop_bottom_frac,
            route_overlay=route_overlay, overlay_kwargs=overlay_kwargs,
            feature_cache_tag=feature_cache_tag, use_augmented_camera=use_augmented_camera
        )
        return (_wrap_loader(train_ds, batch_size, num_workers, True, seed),
                _wrap_loader(val_ds, batch_size, num_workers, False, seed))

    n = len(train_ds)
    if (val_split <= 0.0 and num_folds <= 1) or n < 4:
        return _wrap_loader(train_ds, batch_size, num_workers, True, seed), None

    train_idx, val_idx, groups, keys = route_group_split(
        train_ds, val_split, split_seed, fold=fold, num_folds=num_folds)

    if not val_idx or not train_idx:
        return _wrap_loader(train_ds, batch_size, num_workers, True, seed), None

    fold_note = f", fold {fold}/{num_folds}" if num_folds > 1 else ""
    print(f"--> Validation split: {len(train_idx)} train / {len(val_idx)} val frames "
          f"across {len(keys)} route group(s), seed {split_seed}{fold_note}."
          + ("" if len(keys) > 1 else "  [Warning] Only one group found - this is a"
             " frame-level split and will read optimistically."))

    return (_wrap_loader(Subset(train_ds, train_idx), batch_size, num_workers, True, seed),
            _wrap_loader(Subset(train_ds, val_idx), batch_size, num_workers, False, seed))
