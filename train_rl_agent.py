import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from carla_gym_env import CarlaGymEnv
from camera_easycarla_env import CameraEasyCarlaEnv

# --- PyTorch Neural Network Architectures ---

class PretrainedVisionFeatureExtractor(nn.Module):
    """
    ImageNet Pretrained ResNet Feature Extractor for 256x256 RGB Images + Speed State.
    Supports backbone freezing for fast, sample-efficient RL training.
    """
    def __init__(self, backbone_name="resnet18", features_dim=512, freeze_backbone=True, weights_path=None):
        super(PretrainedVisionFeatureExtractor, self).__init__()
        self.freeze_backbone = freeze_backbone

        if backbone_name == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if weights_path is None else None)
            backbone_out_dim = 512
        else:
            # Default: ResNet18
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights_path is None else None)
            backbone_out_dim = 512

        # Remove the final 1000-class FC classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # -> Outputs (N, 512, 1, 1)

        # Optionally load CARLA-domain pretrained checkpoint (.pth)
        if weights_path is not None and os.path.exists(weights_path):
            print(f"--> Loading CARLA-domain pretrained vision weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            self.backbone.load_state_dict(state_dict, strict=False)

        # Freeze backbone parameters if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ImageNet normalization statistics
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Linear projector for visual vector + speed scalar
        self.fc = nn.Sequential(
            nn.Linear(backbone_out_dim + 1, features_dim),
            nn.ReLU()
        )

    def forward(self, image, speed):
        # Normalize image pixel values [0, 255] -> [0, 1]
        img_x = image.float() / 255.0
        # Permute (N, H, W, C) -> (N, C, H, W)
        if img_x.ndim == 4 and img_x.shape[-1] == 3:
            img_x = img_x.permute(0, 3, 1, 2)

        # Standard ImageNet normalization
        img_normalized = (img_x - self.mean) / self.std

        if self.freeze_backbone:
            with torch.no_grad():
                conv_out = self.backbone(img_normalized).flatten(start_dim=1)
        else:
            conv_out = self.backbone(img_normalized).flatten(start_dim=1)

        speed_x = speed.float().view(-1, 1) / 50.0 # Normalize speed by 50 km/h scale
        combined = torch.cat([conv_out, speed_x], dim=1)
        return self.fc(combined)

class CNNFeatureExtractor(nn.Module):
    """NatureCNN-style architecture for extracting features from 256x256 RGB images + speed state."""
    def __init__(self, in_channels=3, features_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), # -> (32, 63, 63)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          # -> (64, 30, 30)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),          # -> (64, 14, 14)
            nn.ReLU(),
            nn.Flatten()                                          # -> 64 * 14 * 14 = 12544
        )
        
        # Linear projector for flattened image features + speed scalar
        self.fc = nn.Sequential(
            nn.Linear(12544 + 1, features_dim),
            nn.ReLU()
        )

    def forward(self, image, speed):
        # Normalize image pixel values [0, 255] -> [0, 1]
        img_x = image.float() / 255.0
        # Permute (N, H, W, C) -> (N, C, H, W) if needed
        if img_x.ndim == 4 and img_x.shape[-1] == 3:
            img_x = img_x.permute(0, 3, 1, 2)
            
        conv_out = self.conv(img_x)
        speed_x = speed.float().view(-1, 1) / 50.0  # Normalize speed by 50 km/h scale
        combined = torch.cat([conv_out, speed_x], dim=1)
        return self.fc(combined)

class ActorCriticPPO(nn.Module):
    """PPO Actor-Critic Policy Network for Continuous Driving Control."""
    def __init__(self, action_dim=3, features_dim=512, backbone_name="resnet18", freeze_backbone=True, use_pretrained=True, weights_path=None):
        super(ActorCriticPPO, self).__init__()
        
        if use_pretrained:
            self.encoder = PretrainedVisionFeatureExtractor(
                backbone_name=backbone_name,
                features_dim=features_dim,
                freeze_backbone=freeze_backbone,
                weights_path=weights_path
            )
        else:
            self.encoder = CNNFeatureExtractor(in_channels=3, features_dim=features_dim)

        # Actor Head: Outputs mean action [throttle/brake or steer]
        self.actor_mean = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        # Learned Log Standard Deviation for continuous action exploration
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic Head: Outputs scalar state-value V(s)
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_action_and_value(self, image, speed, action=None):
        features = self.encoder(image, speed)
        action_mean = self.actor_mean(features)
        
        # Clip log_std to maintain numerical stability [-2, 0.5]
        action_std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 0.5))
        dist = Normal(action_mean, action_std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(features).squeeze(-1)
        
        return action, log_prob, entropy, value

# --- Main Training Function ---

def train():
    parser = argparse.ArgumentParser(description="Train PPO Deep RL Agent in CARLA Simulator.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--env-type", type=str, default="camera_easycarla", choices=["camera_easycarla", "carla_gym"], help="Environment type")
    default_carla_weights = "/workspace/pretrained_carla/model_0030_0.pth" if os.path.exists("/workspace/pretrained_carla/model_0030_0.pth") else None
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34"], help="Pretrained vision backbone")
    parser.add_argument("--weights-path", type=str, default=default_carla_weights, help="Path to custom CARLA pretrained vision checkpoint (.pth)")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze vision backbone parameters")
    parser.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone", help="Fine-tune vision backbone parameters")
    parser.add_argument("--use-pretrained", action="store_true", default=True, help="Use pretrained vision backbone")
    parser.add_argument("--no-pretrained", action="store_false", dest="use_pretrained", help="Train CNN from scratch")

    parser.add_argument("--total-steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--rollout-steps", type=int, default=250, help="Steps per PPO rollout buffer")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO optimization epochs per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping coefficient")
    parser.add_argument("--log-dir", type=str, default="/workspace/runs", help="TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/checkpoints", help="Model checkpoint directory")
    
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==============================================================")
    print(f"   🚀 Starting Camera-Only PPO Deep RL Training               ")
    print(f"==============================================================")
    print(f"Device: {device} | Environment: {args.env_type}")
    print(f"Vision Backbone: {args.backbone.upper()} (Pretrained: {args.use_pretrained}, Frozen: {args.freeze_backbone})")
    if args.weights_path:
        print(f"CARLA Pretrained Checkpoint: {os.path.abspath(args.weights_path)}")
    print(f"Total Steps: {args.total_steps} | Rollout Buffer: {args.rollout_steps}")
    print(f"TensorBoard Logs: {os.path.abspath(args.log_dir)}")

    writer = SummaryWriter(args.log_dir)
    
    # Initialize Selected Environment
    if args.env_type == "camera_easycarla":
        easy_params = {
            'number_of_vehicles': 10,
            'number_of_walkers': 0,
            'dt': 0.05,
            'ego_vehicle_filter': 'vehicle.tesla.model3',
            'surrounding_vehicle_spawned_randomly': True,
            'port': args.port,
            'town': 'Town10HD_Opt',
            'max_time_episode': args.rollout_steps,
            'max_waypoints': 12,
            'visualize_waypoints': False,
            'desired_speed': 8,
            'max_ego_spawn_times': 200,
            'view_mode': 'top',
            'traffic': 'off',
            'lidar_max_range': 50.0,
            'max_nearby_vehicles': 5,
            'img_width': 256,
            'img_height': 256,
        }
        env = CameraEasyCarlaEnv(params=easy_params)
    else:
        env = CarlaGymEnv(host=args.host, port=args.port, img_width=256, img_height=256, max_steps=args.rollout_steps)
    
    # Initialize PPO Policy & Optimizer
    agent = ActorCriticPPO(
        action_dim=3,
        features_dim=512,
        backbone_name=args.backbone,
        freeze_backbone=args.freeze_backbone,
        use_pretrained=args.use_pretrained,
        weights_path=args.weights_path
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    obs, _ = env.reset()
    global_step = 0
    best_episode_reward = -float("inf")

    episode_rewards = []
    episode_speeds = []
    current_ep_reward = 0
    current_ep_speeds = []

    while global_step < args.total_steps:
        # Storage buffers for Rollout
        obs_images = []
        obs_speeds = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        # 1. Collect Rollout Trajectory
        for step in range(args.rollout_steps):
            global_step += 1
            
            img_tensor = torch.tensor(obs["image"].copy(), dtype=torch.uint8).unsqueeze(0).to(device)
            spd_tensor = torch.tensor(obs["speed"], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(img_tensor, spd_tensor)

            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            obs_images.append(img_tensor.squeeze(0))
            obs_speeds.append(spd_tensor.squeeze(0))
            actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze(0))

            obs = next_obs
            current_ep_reward += reward
            speed_val = info.get("speed_kmh", obs["speed"][0])
            current_ep_speeds.append(speed_val)

            if done:
                episode_rewards.append(current_ep_reward)
                avg_speed = np.mean(current_ep_speeds)
                episode_speeds.append(avg_speed)

                collided_status = info.get("is_collision", info.get("has_collided", False))
                print(f"[Step {global_step:04d}/{args.total_steps}] Episode Finished | Reward: {current_ep_reward:+.2f} | Avg Speed: {avg_speed:.1f} km/h | Collided: {collided_status}")
                writer.add_scalar("Reward/Episode", current_ep_reward, global_step)
                writer.add_scalar("Speed/Avg_kmh", avg_speed, global_step)

                if current_ep_reward > best_episode_reward:
                    best_episode_reward = current_ep_reward
                    checkpoint_path = os.path.join(args.checkpoint_dir, "ppo_carla_best.pth")
                    torch.save(agent.state_dict(), checkpoint_path)
                    print(f"--> Saved new BEST policy checkpoint: {checkpoint_path}")

                obs, _ = env.reset()
                current_ep_reward = 0
                current_ep_speeds = []

        # 2. Compute Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_img = torch.tensor(obs["image"].copy(), dtype=torch.uint8).unsqueeze(0).to(device)
            next_spd = torch.tensor(obs["speed"], dtype=torch.float32).unsqueeze(0).to(device)
            next_val = agent.get_action_and_value(next_img, next_spd)[3].squeeze(0)

        returns = []
        advantages = []
        gae = 0
        for t in reversed(range(args.rollout_steps)):
            if t == args.rollout_steps - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = next_val
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = values[t + 1]

            delta = rewards[t] + args.gamma * next_value * next_non_terminal - values[t]
            gae = delta + args.gamma * args.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        # Convert Rollout Lists to Tensors
        b_images = torch.stack(obs_images)
        b_speeds = torch.stack(obs_speeds)
        b_actions = torch.stack(actions)
        b_log_probs = torch.stack(log_probs)
        b_advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        b_returns = torch.tensor(returns, dtype=torch.float32).to(device)
        b_values = torch.stack(values)

        # Normalize Advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # 3. Optimize PPO Policy & Value Networks
        policy_losses = []
        value_losses = []

        for epoch in range(args.ppo_epochs):
            _, new_log_prob, entropy, new_value = agent.get_action_and_value(b_images, b_speeds, b_actions)
            logratio = new_log_prob - b_log_probs
            ratio = logratio.exp()

            # PPO Clipped Surrogate Loss
            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value Loss
            v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()

            # Combined Total Loss
            total_loss = pg_loss + 0.5 * v_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optimizer.step()

            policy_losses.append(pg_loss.item())
            value_losses.append(v_loss.item())

        writer.add_scalar("Loss/Policy", np.mean(policy_losses), global_step)
        writer.add_scalar("Loss/Value", np.mean(value_losses), global_step)
        print(f"--- Rollout Update Complete | Step: {global_step}/{args.total_steps} | Policy Loss: {np.mean(policy_losses):.4f} | Value Loss: {np.mean(value_losses):.4f} ---")

        # Save Latest Checkpoint
        latest_path = os.path.join(args.checkpoint_dir, "ppo_carla_latest.pth")
        torch.save(agent.state_dict(), latest_path)

    env.close()
    writer.close()
    print("--- Training Completed Successfully! ---")
    print(f"Final Model Saved to: {os.path.abspath(args.checkpoint_dir)}")

if __name__ == "__main__":
    train()
