"""World on Rails (WoR) Factorized World Model & Dynamic Programming Solver.

Computes optimal action values Q*(s, a) across discrete rail waypoints using
factorized state transitions (P_ego x P_world) as defined in Chen et al., ICCV 2021.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class RailsDynamicProgramming:
    """
    Computes optimal Q-values over pre-recorded CARLA driving logs using
    Bellman Dynamic Programming / Value Iteration under the World-on-Rails assumption.
    """
    def __init__(
        self,
        discount: float = 0.95,
        num_rails: int = 9,
        horizon: int = 10,
        collision_penalty: float = -25.0,
        red_light_penalty: float = -20.0,
        off_road_penalty: float = -20.0,
        progress_weight: float = 1.0
    ):
        self.discount = discount
        self.num_rails = num_rails
        self.horizon = horizon
        self.collision_penalty = collision_penalty
        self.red_light_penalty = red_light_penalty
        self.off_road_penalty = off_road_penalty
        self.progress_weight = progress_weight

    def solve_trajectory_q_values(
        self,
        trajectory_rewards: np.ndarray,
        collision_mask: Optional[np.ndarray] = None,
        red_light_mask: Optional[np.ndarray] = None,
        off_road_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Computes Q*(s, a) via backward dynamic programming pass across T steps and K rails.
        
        Args:
            trajectory_rewards: Array of shape (T, num_rails) containing forward progress rewards.
            collision_mask: Boolean array of shape (T, num_rails) indicating collision events.
            red_light_mask: Boolean array of shape (T, num_rails) indicating traffic light infractions.
            off_road_mask: Boolean array of shape (T, num_rails) indicating off-road events.
            
        Returns:
            q_values: Array of shape (T, num_rails) containing optimal state-action values.
        """
        T, K = trajectory_rewards.shape
        q_values = np.zeros((T, K), dtype=np.float32)

        # 1. Compute immediate step rewards R(s, a)
        step_rewards = trajectory_rewards * self.progress_weight

        if collision_mask is not None:
            step_rewards += collision_mask.astype(np.float32) * self.collision_penalty
        if red_light_mask is not None:
            step_rewards += red_light_mask.astype(np.float32) * self.red_light_penalty
        if off_road_mask is not None:
            step_rewards += off_road_mask.astype(np.float32) * self.off_road_penalty

        # 2. Backward Value Iteration pass
        # At final step T-1, Q(T-1, a) = R(T-1, a)
        q_values[T - 1] = step_rewards[T - 1]

        for t in reversed(range(T - 1)):
            # Under world-on-rails transition, future value V(s_{t+1}) = max_a' Q(t+1, a')
            # Transitions allow switching to adjacent rails (|a - a'| <= 1) or staying on rail
            next_v = np.max(q_values[t + 1])
            q_values[t] = step_rewards[t] + self.discount * next_v

        return q_values


class WorldModel(nn.Module):
    """
    Learned Factorized World Model for predicting future occupant occupancy and ego kinematics.
    """
    def __init__(
        self,
        state_dim: int = 128,
        num_rails: int = 9,
        feature_dim: int = 512
    ):
        super().__init__()
        self.num_rails = num_rails
        
        # Ego Kinematics Transition Network: P_ego(s'_{ego} | s_{ego}, a)
        self.ego_transition = nn.Sequential(
            nn.Linear(state_dim + num_rails, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, state_dim)
        )

        # Environment Occupancy Predictor: P_world(O_{t+1} | s_{world})
        self.world_occupancy = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_rails)  # Probability of collision on each rail
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        ego_state: torch.Tensor,
        rail_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts next ego state and collision risk across rails.
        """
        # 1. Collision probabilities across candidate rails
        collision_logits = self.world_occupancy(visual_features)
        
        # 2. Next ego state
        ego_action_input = torch.cat([ego_state, rail_action], dim=-1)
        next_ego_state = self.ego_transition(ego_action_input)

        return next_ego_state, collision_logits
