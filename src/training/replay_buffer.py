"""Off-policy replay buffer storing cached visual features for sample-efficient SAC updates."""
from typing import Tuple, Union
import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer over cached visual embeddings.

    Because the vision backbone is frozen, transitions are stored as the backbone's output
    (visual_dim floats) rather than raw 256x768x3 frames - roughly a 1000x memory saving,
    and the encoder never re-runs during gradient updates.

    Rewards are stored RAW (only clipped). Normalizing them with a running statistic would
    be a correctness bug here: transitions written early would carry a different reward
    scale than transitions written later, and the Bellman backup mixes them freely.
    """
    def __init__(
        self,
        capacity: int,
        visual_dim: int,
        action_dim: int = 3,
        device: torch.device = None,
        store_on_gpu: bool = False
    ):
        self.capacity = int(capacity)
        self.visual_dim = int(visual_dim)
        self.action_dim = int(action_dim)
        self.device = device or torch.device("cpu")
        # Feature tensors dominate memory; keep them on CPU unless explicitly told otherwise.
        store_device = self.device if store_on_gpu else torch.device("cpu")
        self.store_device = store_device

        self.visual = torch.zeros((self.capacity, self.visual_dim), dtype=torch.float32, device=store_device)
        self.next_visual = torch.zeros((self.capacity, self.visual_dim), dtype=torch.float32, device=store_device)
        self.speed = torch.zeros((self.capacity, 1), dtype=torch.float32, device=store_device)
        self.next_speed = torch.zeros((self.capacity, 1), dtype=torch.float32, device=store_device)
        self.action = torch.zeros((self.capacity, self.action_dim), dtype=torch.float32, device=store_device)
        self.reward = torch.zeros((self.capacity,), dtype=torch.float32, device=store_device)
        self.done = torch.zeros((self.capacity,), dtype=torch.float32, device=store_device)

        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    @property
    def memory_mb(self) -> float:
        """Approximate resident size of the stored tensors in megabytes."""
        per_row = (2 * self.visual_dim + 2 + self.action_dim + 2) * 4
        return (self.capacity * per_row) / (1024.0 ** 2)

    def add(
        self,
        visual: torch.Tensor,
        speed: torch.Tensor,
        action: torch.Tensor,
        reward: Union[np.ndarray, torch.Tensor],
        next_visual: torch.Tensor,
        next_speed: torch.Tensor,
        done: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Insert one timestep from every parallel environment, wrapping at capacity."""
        n = visual.shape[0]
        rew_t = torch.as_tensor(reward, dtype=torch.float32).view(n)
        done_t = torch.as_tensor(done, dtype=torch.float32).view(n)

        for i in range(n):
            idx = self.pos
            self.visual[idx] = visual[i].detach().to(self.store_device, torch.float32)
            self.next_visual[idx] = next_visual[i].detach().to(self.store_device, torch.float32)
            self.speed[idx] = speed[i].detach().to(self.store_device, torch.float32).view(1)
            self.next_speed[idx] = next_speed[i].detach().to(self.store_device, torch.float32).view(1)
            self.action[idx] = action[i].detach().to(self.store_device, torch.float32)
            self.reward[idx] = rew_t[i]
            self.done[idx] = done_t[i]

            self.pos += 1
            if self.pos >= self.capacity:
                self.pos = 0
                self.full = True

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Draw a uniform random batch, moved to the compute device."""
        size = len(self)
        if size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        idx = torch.randint(0, size, (min(batch_size, size),), device=self.store_device)

        return (
            self.visual[idx].to(self.device, non_blocking=True),
            self.speed[idx].to(self.device, non_blocking=True),
            self.action[idx].to(self.device, non_blocking=True),
            self.reward[idx].to(self.device, non_blocking=True),
            self.next_visual[idx].to(self.device, non_blocking=True),
            self.next_speed[idx].to(self.device, non_blocking=True),
            self.done[idx].to(self.device, non_blocking=True)
        )
