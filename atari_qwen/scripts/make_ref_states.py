"""Save the 192 reference states of mixer_diag.states() (qnowd seed-1 replay, second half, rng seed 0) as uint8 for the box watcher."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mixer_diag as md
obs, act = md.states()
u8 = (obs * 255.0).round().clamp(0, 255).to(dtype=__import__("torch").uint8).numpy()
assert np.array_equal(u8.astype(np.float32) / 255.0, obs.numpy())
np.savez(os.path.join(HERE, "ref_states_qnowd_s1.npz"), obs=u8, act=act.numpy())
print("saved", u8.shape, act.shape)
