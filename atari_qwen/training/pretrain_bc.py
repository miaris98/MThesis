"""Behavioral Cloning (BC) Pretraining on Expert Atari Demonstrations for Qwen Transformer."""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from atari_qwen.config.atari_config import get_config
from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic
from atari_qwen.training.train_ppo import evaluate_policy


def pretrain_bc(
    data_path: str = "data/atari_expert/breakout_expert_100k.npz",
    env_id: str = "BreakoutNoFrameskip-v4",
    preset: str = "tiny",
    encoder_type: str = "impala_cnn",
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    save_path: str = "results/atari_qwen/checkpoints/qwen_bc_pretrained.pt",
    device_str: str = "auto"
) -> str:
    """Pretrain Qwen Actor-Critic on offline expert dataset via cross-entropy Behavioral Cloning."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print("================================================================")
    print("--> Starting Behavioral Cloning Pretraining on Expert Data")
    print(f"--> Dataset:    {data_path}")
    print(f"--> Model:      Qwen-{preset.capitalize()} + {encoder_type.upper()}")
    print(f"--> Epochs:     {epochs} | Batch Size: {batch_size} | Device: {device}")
    print(f"--> Output:     {save_path}")
    print("================================================================\n")

    # 1. Load dataset
    print(f"--> Loading expert transitions from {data_path}...")
    data = np.load(data_path)
    obs = data["obs"]
    actions = data["actions"]
    print(f"✓ Loaded {len(obs):,} transitions (obs shape: {obs.shape}, actions shape: {actions.shape})")

    # 2. Validation split (10%)
    val_split = int(0.10 * len(obs))
    train_obs = torch.from_numpy(obs[:-val_split])
    train_act = torch.from_numpy(actions[:-val_split]).long()
    val_obs = torch.from_numpy(obs[-val_split:])
    val_act = torch.from_numpy(actions[-val_split:]).long()

    train_loader = DataLoader(TensorDataset(train_obs, train_act), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(TensorDataset(val_obs, val_act), batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # 3. Model & Optimizer
    model = QwenAtariActorCritic(
        action_dim=4,
        in_channels=4,
        preset=preset,
        encoder_type=encoder_type
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for b_obs, b_act in train_loader:
            b_obs = b_obs.to(device)
            b_act = b_act.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                actor_repr, _ = model._forward_transformer(b_obs)
                logits = model.actor_head(actor_repr)
                loss = criterion(logits, b_act)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(b_act)
            train_correct += (logits.argmax(dim=-1) == b_act).sum().item()
            total_train += len(b_act)

        train_acc = (train_correct / total_train) * 100.0
        train_loss /= total_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for b_obs, b_act in val_loader:
                b_obs = b_obs.to(device)
                b_act = b_act.to(device)
                actor_repr, _ = model._forward_transformer(b_obs)
                logits = model.actor_head(actor_repr)
                loss = criterion(logits, b_act)
                val_loss += loss.item() * len(b_act)
                val_correct += (logits.argmax(dim=-1) == b_act).sum().item()
                total_val += len(b_act)

        val_acc = (val_correct / total_val) * 100.0
        val_loss /= total_val

        elapsed = time.time() - start_time
        fps = total_train / (elapsed / epoch)
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}% | Speed: {fps:5.0f} samples/s")

    # Evaluate pretrained policy live in Atari
    print("\n--> Evaluating Pretrained Qwen Policy in Atari Environment...")
    eval_score, eval_std = evaluate_policy(
        model=model,
        env_id=env_id,
        device=device,
        num_episodes=5,
        seed=42
    )
    print(f"✓ Pretrained Policy Zero-Shot Score: {eval_score:.1f} +/- {eval_std:.1f} (Human Benchmark: 30.5)")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "preset": preset,
        "encoder_type": encoder_type,
        "eval_score": eval_score,
        "val_acc": val_acc
    }, save_path)
    print(f"\n✓ Saved Pretrained Checkpoint: {save_path}\n")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Cloning pretraining on expert data")
    parser.add_argument("--data-path", type=str, default="data/atari_expert/breakout_expert_100k.npz")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--preset", type=str, default="tiny")
    parser.add_argument("--encoder-type", type=str, default="impala_cnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str, default="results/atari_qwen/checkpoints/qwen_bc_pretrained.pt")
    args = parser.parse_args()

    pretrain_bc(
        data_path=args.data_path,
        env_id=args.env_id,
        preset=args.preset,
        encoder_type=args.encoder_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )
