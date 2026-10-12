import gymnasium as gym
import numpy as np
import torch
from atari_qwen.envs.atari_wrappers import make_atari_env
from atari_qwen.data.generate_expert_data import load_expert_dqn

def test_expert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert = load_expert_dqn(device)
    
    # Test with standard env
    env_fn = make_atari_env("BreakoutNoFrameskip-v4", seed=42, idx=0, noop_max=0, clip_reward=False, episodic_life=False)
    env = env_fn()
    obs, _ = env.reset()
    
    # Check what expert predicts on frame 0
    obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
    with torch.no_grad():
        q_vals = expert(obs_t)
        act = torch.argmax(q_vals, dim=-1).item()
    print("Initial expert action:", act, "Q values:", q_vals.cpu().numpy().tolist())
    
    # Run 1000 steps with expert
    score = 0
    lives = 5
    actions_taken = []
    for step in range(1000):
        # Check lives
        curr_lives = env.unwrapped.ale.lives()
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = expert(obs_t)
            act = torch.argmax(q_vals, dim=-1).item()
            
        # If ball unserved (lives dropped or initial)
        if step == 0 or curr_lives < lives:
            act = 1 # FIRE
        lives = curr_lives
        
        actions_taken.append(act)
        obs, r, term, trunc, _ = env.step(act)
        score += r
        if term or trunc:
            print(f"Terminated at step {step}, score={score}")
            break
            
    print(f"Total steps: {len(actions_taken)}, Score: {score}")
    print("Action counts [0: NOOP, 1: FIRE, 2: RIGHT, 3: LEFT]:", np.bincount(actions_taken, minlength=4))

if __name__ == "__main__":
    test_expert()
