import os
import sys
import warnings

# Completely silence runtime, deprecation, and future warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

try:
    _old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    import gym
    if hasattr(gym, 'logger'):
        gym.logger.set_level(40)
except Exception:
    pass
finally:
    sys.stderr = _old_stderr

# Prevent OpenBLAS/MKL/OMP thread explosion on high-core CPUs (AMD EPYC)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Ensure project root is in Python module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.training_config import TrainingConfig
from src.training.ppo_trainer import PPOTrainer
from src.models.actor_critic import ActorCriticPPO  # Re-export for external tools / evaluation


def main():
    """Parse CLI flags and execute the selected training engine (PPO or SAC)."""
    config = TrainingConfig.from_args()
    if config.algo == "sac":
        # Imported lazily so a PPO run never pays for the SAC module graph.
        from src.training.sac_trainer import SACTrainer
        trainer = SACTrainer(config)
    else:
        trainer = PPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
