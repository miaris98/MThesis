import numpy as np
d = np.load("/root/MThesis/data/atari_expert/breakout_expert_100k.npz")
bc = np.bincount(d['actions'])
print("Action distribution [NOOP, FIRE, RIGHT, LEFT]:", bc.tolist())
print("Reward sum:", d['rewards'].sum())
print("Nonzero rewards:", int((d['rewards'] > 0).sum()))
print("Dones count:", int(d['dones'].sum()))
