"""Training package containing RolloutBuffer, PPOTrainer, and WorldOnRailsTrainer.

`PPOTrainer` and `RolloutBuffer` are lazy-loaded via `__getattr__` (PEP 562) rather
than imported at module top-level. `ppo_trainer` imports `src.envs.vector_carla_env`,
which imports CARLA's compiled Python bindings - a native extension that has been
observed to segfault the interpreter on import on at least one vast.ai image/driver
combination. A segfault is a process-level crash, not a Python exception, so no amount
of try/except at import time can recover from it; the only fix is to not import that
chain at all for code that never uses it. The offline WoR path (`wor_trainer`,
`wor_dataset`, and everything `train_wor.py` needs) has no CARLA dependency and must
stay importable even when the online CARLA RL path cannot be imported on the host.

No code in this project imports `PPOTrainer`/`RolloutBuffer` via `from src.training
import ...` - every consumer imports directly from `src.training.ppo_trainer` /
`src.training.rollout_buffer` - so this is a pure safety net, not a behavior change.

`WorldOnRailsTrainer` and the dataset helpers are lazy for the same reason (2026-11-02):
the closed-loop agent imports `src.training.counterfactual_q` (numpy only) for the A71
lane shift, and an eager trainer import pulled in src.logging -> tensorboard, which the
CARLA eval venv does not have. A71-on crashed routes on its first confirmed lane change.
"""
_LAZY = {"WorldOnRailsTrainer": "src.training.wor_trainer", "WorldOnRailsDataset": "src.training.wor_dataset",
         "create_wor_dataloader": "src.training.wor_dataset"}

__all__ = [
    "RolloutBuffer",
    "PPOTrainer",
    "WorldOnRailsTrainer",
    "WorldOnRailsDataset",
    "create_wor_dataloader"
]


def __getattr__(name: str):
    if name == "PPOTrainer":
        from src.training.ppo_trainer import PPOTrainer
        return PPOTrainer
    if name == "RolloutBuffer":
        from src.training.rollout_buffer import RolloutBuffer
        return RolloutBuffer
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
