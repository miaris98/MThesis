"""Monte Carlo Tree Search (MCTS) Engine for Latent Dynamics Models (MuZero / EfficientZero style).

Unrolls search trees purely in latent space using `agent.predictor.predict_step(z, a)`:
1. Selection via Min-Max Normalized PUCT
2. Latent Expansion & Evaluation via the learned dynamics model and value/reward heads
3. Backpropagation of discounted rewards and terminal value estimates
4. Dirichlet noise at root for exploration during data collection
"""
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MinMaxStats:
    """Tracks min and max values seen in a search tree to normalize Q-values into [0, 1]."""
    def __init__(self, known_bounds: Optional[Tuple[float, float]] = None):
        self.maximum = known_bounds[1] if known_bounds else -float('inf')
        self.minimum = known_bounds[0] if known_bounds else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTSNode:
    """A node in the MCTS latent search tree."""
    def __init__(self, prior: float):
        self.prior: float = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.latent_z: Optional[torch.Tensor] = None
        self.reward: float = 0.0
        self.is_expanded: bool = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSEngine:
    """Latent Monte Carlo Tree Search Engine."""
    def __init__(
        self,
        action_dim: int = 4,
        num_simulations: int = 35,
        discount: float = 0.99,
        c_puct_base: float = 19652.0,
        c_puct_init: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ):
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.discount = discount
        self.c_puct_base = c_puct_base
        self.c_puct_init = c_puct_init
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def _select_child(self, node: MCTSNode, min_max_stats: MinMaxStats) -> Tuple[int, MCTSNode]:
        """Select child action maximizing the PUCT score."""
        # Unvisited children exploration guarantee: visit each child action at least once
        for action, child in node.children.items():
            if child.visit_count == 0:
                return action, child

        best_score = -float('inf')
        best_action = -1
        best_child = None

        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)

        for action, child in node.children.items():
            # Normalized Q-value
            q_value = min_max_stats.normalize(child.reward + self.discount * child.value)

            # PUCT exploration bonus
            pb_c = math.log((total_visits + self.c_puct_base + 1.0) / self.c_puct_base) + self.c_puct_init
            pb_c *= sqrt_total / (child.visit_count + 1)
            u_value = pb_c * child.prior

            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def search_single(
        self,
        root_latent: torch.Tensor,
        agent: nn.Module,
        device: torch.device,
        root_policy_repr: Optional[torch.Tensor] = None,
        add_dirichlet: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, int, float]:
        """Runs MCTS from a single root latent state z_0.

        Args:
            root_latent: (latent_dim,) root latent state
            agent: ImpalaGTrXLAgent or similar with actor_head, critic_head, predictor
            device: torch device
            root_policy_repr: optional (embed_dim,) policy representation for actor head
            add_dirichlet: whether to add Dirichlet noise to root prior
            temperature: visit count temperature (0.0 for argmax)

        Returns:
            (visit_policy_probs, chosen_action, root_value)
        """
        root = MCTSNode(prior=1.0)
        root.latent_z = root_latent.unsqueeze(0)  # (1, latent_dim)
        min_max_stats = MinMaxStats()

        # 1. Expand root using policy and critic heads
        with torch.no_grad():
            pol_in = root_policy_repr.unsqueeze(0) if root_policy_repr is not None else root.latent_z
            policy_logits = agent.actor_head(pol_in)
            priors = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            root_val = agent.critic_head(root.latent_z).item()

        min_max_stats.update(root_val)

        # Apply Dirichlet exploration noise at root if training
        if add_dirichlet and self.dirichlet_eps > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * self.action_dim)
            priors = (1 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise

        for a in range(self.action_dim):
            root.children[a] = MCTSNode(prior=float(priors[a]))
        root.is_expanded = True

        # 2. Run MCTS Simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            action_path = []

            # A. Traverse tree until leaf
            while node.is_expanded and node.children:
                action, node = self._select_child(node, min_max_stats)
                search_path.append(node)
                action_path.append(action)

            parent = search_path[-2]
            action_taken = action_path[-1]

            # B. Expand leaf using latent dynamics model: z_{k+1}, r_hat, v_hat
            with torch.no_grad():
                act_t = torch.tensor([action_taken], dtype=torch.long, device=device)
                z_next, r_hat, v_hat = agent.predictor.predict_step(parent.latent_z, act_t)
                
                leaf_value = v_hat.item()
                node.reward = r_hat.item()
                node.latent_z = z_next

                # Expand children of leaf
                leaf_logits = agent.actor_head(z_next)
                leaf_priors = F.softmax(leaf_logits, dim=-1).squeeze(0).cpu().numpy()

                for a in range(self.action_dim):
                    node.children[a] = MCTSNode(prior=float(leaf_priors[a]))
                node.is_expanded = True

            min_max_stats.update(leaf_value)

            # C. Backpropagation
            # Propagate value up the path: G_t = r_{t+1} + gamma * G_{t+1}
            value = leaf_value
            for i in reversed(range(len(search_path))):
                curr_node = search_path[i]
                curr_node.visit_count += 1
                curr_node.value_sum += value
                value = curr_node.reward + self.discount * value

        # 3. Compute visit count distribution
        visits = np.array([root.children[a].visit_count for a in range(self.action_dim)], dtype=np.float32)
        if temperature == 0.0:
            probs = np.zeros(self.action_dim, dtype=np.float32)
            max_visits = np.max(visits)
            candidates = [a for a in range(self.action_dim) if visits[a] == max_visits]
            if len(candidates) == 1:
                chosen_action = candidates[0]
            else:
                q_vals = [root.children[a].value for a in candidates]
                chosen_action = candidates[int(np.argmax(q_vals))]
            probs[chosen_action] = 1.0
        else:
            visits_temp = visits ** (1.0 / max(temperature, 1e-4))
            probs = visits_temp / np.sum(visits_temp)
            chosen_action = int(np.random.choice(self.action_dim, p=probs))

        return probs, chosen_action, root.value

    def search_batch(
        self,
        root_latents: torch.Tensor,
        agent: nn.Module,
        device: torch.device,
        root_policy_reprs: Optional[torch.Tensor] = None,
        add_dirichlet: bool = True,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs MCTS across a batch of environments with batched GPU evaluation.

        Args:
            root_latents: (B, latent_dim)
            root_policy_reprs: optional (B, embed_dim)
        Returns:
            (batch_probs, batch_actions, batch_values)
        """
        B = root_latents.shape[0]
        if B == 0:
            return np.zeros((0, self.action_dim), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        roots = [MCTSNode(prior=1.0) for _ in range(B)]
        min_max_stats_list = [MinMaxStats() for _ in range(B)]

        for b in range(B):
            roots[b].latent_z = root_latents[b : b + 1]

        # 1. Expand all roots in one batched forward pass
        with torch.no_grad():
            pol_in = root_policy_reprs if root_policy_reprs is not None else root_latents
            policy_logits = agent.actor_head(pol_in)
            priors_batch = F.softmax(policy_logits, dim=-1).cpu().numpy()  # (B, action_dim)
            root_vals = agent.critic_head(root_latents).squeeze(-1).cpu().numpy()

        for b in range(B):
            r_val = float(root_vals[b]) if np.ndim(root_vals) > 0 else float(root_vals)
            min_max_stats_list[b].update(r_val)
            priors = priors_batch[b]
            if add_dirichlet and self.dirichlet_eps > 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * self.action_dim)
                priors = (1 - self.dirichlet_eps) * priors + self.dirichlet_eps * noise

            for a in range(self.action_dim):
                roots[b].children[a] = MCTSNode(prior=float(priors[a]))
            roots[b].is_expanded = True

        # 2. Run MCTS simulations with batched GPU expansion
        for _ in range(self.num_simulations):
            search_paths = []
            parents = []
            actions_taken = []
            leaf_nodes = []

            # A. Traverse tree until leaf for each environment
            for b in range(B):
                node = roots[b]
                search_path = [node]
                action_path = []

                while node.is_expanded and node.children:
                    action, node = self._select_child(node, min_max_stats_list[b])
                    search_path.append(node)
                    action_path.append(action)

                search_paths.append(search_path)
                parents.append(search_path[-2])
                actions_taken.append(action_path[-1])
                leaf_nodes.append(node)

            # B. Batched GPU evaluation across all B environments
            batch_parent_z = torch.cat([p.latent_z for p in parents], dim=0)
            batch_act_t = torch.tensor(actions_taken, dtype=torch.long, device=device)

            with torch.no_grad():
                z_next_batch, r_hat_batch, v_hat_batch = agent.predictor.predict_step(batch_parent_z, batch_act_t)
                leaf_logits_batch = agent.actor_head(z_next_batch)
                leaf_priors_batch = F.softmax(leaf_logits_batch, dim=-1).cpu().numpy()  # (B, action_dim)
                r_vals = r_hat_batch.cpu().numpy()
                v_vals = v_hat_batch.cpu().numpy()

            # C. Assign and backpropagate for each environment
            for b in range(B):
                leaf = leaf_nodes[b]
                leaf.latent_z = z_next_batch[b : b + 1]
                r_val = float(r_vals[b]) if np.ndim(r_vals) > 0 else float(r_vals)
                leaf_val = float(v_vals[b]) if np.ndim(v_vals) > 0 else float(v_vals)
                leaf.reward = r_val

                for a in range(self.action_dim):
                    leaf.children[a] = MCTSNode(prior=float(leaf_priors_batch[b, a]))
                leaf.is_expanded = True

                min_max_stats_list[b].update(leaf_val)

                value = leaf_val
                for i in reversed(range(len(search_paths[b]))):
                    curr_node = search_paths[b][i]
                    curr_node.visit_count += 1
                    curr_node.value_sum += value
                    value = curr_node.reward + self.discount * value

        # 3. Compute visit distributions and actions
        batch_probs = []
        batch_actions = []
        batch_values = []

        for b in range(B):
            root = roots[b]
            visits = np.array([root.children[a].visit_count for a in range(self.action_dim)], dtype=np.float32)
            if temperature == 0.0:
                probs = np.zeros(self.action_dim, dtype=np.float32)
                max_visits = np.max(visits)
                candidates = [a for a in range(self.action_dim) if visits[a] == max_visits]
                if len(candidates) == 1:
                    chosen_action = candidates[0]
                else:
                    q_vals = [root.children[a].value for a in candidates]
                    chosen_action = candidates[int(np.argmax(q_vals))]
                probs[chosen_action] = 1.0
            else:
                visits_temp = visits ** (1.0 / max(temperature, 1e-4))
                probs = visits_temp / np.sum(visits_temp)
                chosen_action = int(np.random.choice(self.action_dim, p=probs))

            batch_probs.append(probs)
            batch_actions.append(chosen_action)
            batch_values.append(root.value)

        return np.array(batch_probs), np.array(batch_actions), np.array(batch_values)

