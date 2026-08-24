"""Command-line entry point for Vision-Based Deep RL in CARLA."""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure project root is in Python module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.training_config import TrainingConfig
from src.training.ppo_trainer import PPOTrainer
from src.models.actor_critic import ActorCriticPPO  # Re-export for external tools / evaluation


def main():
    """Parse CLI flags and execute PPO training engine."""
    config = TrainingConfig.from_args()
    trainer = PPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
