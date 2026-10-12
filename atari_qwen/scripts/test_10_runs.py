import gymnasium as gym
import numpy as np
import sys
from pathlib import Path

# Add repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from atari_qwen.envs.atari_wrappers import make_atari_env

print("--> Testing Breakout Environment and True Uniform-Random Policy Dynamics...")
env_fn = make_atari_env("BreakoutNoFrameskip-v4", seed=42, idx=0, noop_max=0, clip_reward=False, episodic_life=False)
env = env_fn()
n_actions = env.action_space.n
print(f"Action meanings: {env.unwrapped.get_action_meanings()}")

# Uniform random over ALL actions (including FIRE), with no scripted relaunch-on-life-loss
# or stuck-detection help. This is a fair apples-to-apples baseline for a trained argmax
# policy -- the old version excluded FIRE from the random pool and force-fired on every
# life loss/stall, which made it a scripted heuristic rather than a random policy.
scores = []
for t in range(1, 11):
    obs, _ = env.reset()
    done = False
    ep_ret = 0.0
    steps = 0

    while not done and steps < 3000:
        steps += 1
        action = np.random.randint(0, n_actions)

        step_res = env.step(action)
        if len(step_res) == 5:
            obs, r, term, trunc, _ = step_res
            done = term or trunc
        else:
            obs, r, done, _ = step_res

        ep_ret += r

    scores.append(ep_ret)
    print(f"Test #{t:02d}: Score = {ep_ret:.1f} | Steps = {steps} | Lives remaining = {env.unwrapped.ale.lives()}")

env.close()
print(f"\nSummary of 10 Runs: Mean Score = {np.mean(scores):.2f}, Max Score = {np.max(scores):.1f}")
