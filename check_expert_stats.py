import numpy as np
import torch
from atari_qwen.models.qwen_atari_actor_critic import QwenAtariActorCritic
from atari_qwen.envs.atari_wrappers import make_atari_env

def main():
    d = np.load("/root/MThesis/data/atari_expert/breakout_expert_100k.npz")
    print("=== DATASET STATS ===")
    print("Action counts [0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT]:", np.bincount(d['actions']))
    print("Total reward:", d['rewards'].sum())
    print("Nonzero rewards:", (d['rewards'] > 0).sum())
    print("Dones count:", d['dones'].sum())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("results/atari_qwen/checkpoints/qwen_bc_pretrained.pt", map_location=device)
    model = QwenAtariActorCritic(action_dim=4, in_channels=4, preset="tiny", encoder_type="impala_cnn").to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    obs_batch = torch.from_numpy(d['obs'][:200]).to(device)
    with torch.no_grad():
        repr, _ = model._forward_transformer(obs_batch)
        logits = model.actor_head(repr)
        preds = logits.argmax(dim=-1).cpu().numpy()
    
    print("Model prediction counts on 200 samples:", np.bincount(preds, minlength=4))
    print("First 20 expert actions:   ", d['actions'][:20].tolist())
    print("First 20 predicted actions:", preds[:20].tolist())

if __name__ == "__main__":
    main()
