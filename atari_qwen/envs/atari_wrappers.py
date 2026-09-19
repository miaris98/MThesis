"""Standard Atari 2600 preprocessing wrappers optimized for high-throughput RL."""
from collections import deque
from typing import Callable, Optional, Tuple, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import gymnasium as gym
    from gymnasium import spaces
    try:
        import ale_py
        if hasattr(gym, "register_envs"):
            gym.register_envs(ale_py)
    except ImportError:
        pass
except ImportError:
    import gym
    from gym import spaces


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking a random number (1..noop_max) of no-ops on reset."""
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        
        obs = None
        info = {}
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            if done:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Take FIRE action on reset for environments that require it to begin play."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        assert len(action_meanings) >= 3 and action_meanings[1] == 'FIRE'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        step_result = self.env.step(1)
        if len(step_result) == 5:
            obs, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, _, done, info = step_result
        if done:
            self.env.reset(**kwargs)
        step_result = self.env.step(2)
        if len(step_result) == 5:
            obs, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, _, done, info = step_result
        if done:
            self.env.reset(**kwargs)
        return obs, info


class StickyActionEnv(gym.Wrapper):
    """With probability p, repeat the previous action instead of the requested one (Machado et
    al. 2018 sticky actions) -- breaks the deterministic-emulator looping a constant-action
    policy exploits to reproduce the same score every eval (TODO_GTRXL_COLLAPSE E36)."""
    def __init__(self, env: gym.Env, p: float = 0.25):
        super().__init__(env)
        self.p = p
        self._last_action = 0

    def reset(self, **kwargs):
        self._last_action = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if np.random.random() < self.p:
            action = self._last_action
        self._last_action = action
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            five_tuple = True
        else:
            obs, reward, done, info = step_result
            terminated, truncated = done, False
            five_tuple = False

        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # For value bootstrap, end episode on life loss
            terminated = True
            done = True
        self.lives = lives
        
        if five_tuple:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Resend step with 0 to proceed after losing life
            step_result = self.env.step(0)
            if len(step_result) == 5:
                obs, _, _, _, info = step_result
            else:
                obs, _, _, info = step_result
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame, taking the pixel-wise max over the last 2 frames."""
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = done = False
        info = {}
        for i in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                five_tuple = True
            else:
                obs, reward, done, info = step_result
                terminated, truncated = done, False
                five_tuple = False
            
            total_reward += reward
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        if five_tuple:
            return max_frame, total_reward, terminated, truncated, info
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward to {-1, 0, +1} based on its sign."""
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 grayscale, matching DeepMind DQN / Nature papers."""
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            # Fallback luminance conversion and nearest-neighbor / slice downsampling
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            # Resize via simple indexing if cv2 is not present
            h_indices = np.linspace(0, frame.shape[0] - 1, self.height, dtype=int)
            w_indices = np.linspace(0, frame.shape[1] - 1, self.width, dtype=int)
            frame = frame[h_indices[:, None], w_indices]
        return frame


class FrameStack(gym.Wrapper):
    """Stack k consecutive frames along the channel dimension: (k, 84, 84)."""
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=((k,) + shp), dtype=np.uint8
        )

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) else res
        info = res[1] if isinstance(res, tuple) else {}
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            five_tuple = True
        else:
            obs, reward, done, info = step_result
            terminated, truncated = done, False
            five_tuple = False
        self.frames.append(obs)
        if five_tuple:
            return self._get_ob(), reward, terminated, truncated, info
        return self._get_ob(), reward, terminated or truncated, info

    def _get_ob(self) -> np.ndarray:
        return np.array(self.frames)


def make_atari_env(
    env_id: str,
    seed: int = 0,
    idx: int = 0,
    noop_max: int = 30,
    frame_stack: int = 4,
    clip_reward: bool = True,
    episodic_life: bool = True
) -> Callable[[], gym.Env]:
    """Factory creating a fully wrapped single Atari environment instance."""
    def _thunk() -> gym.Env:
        env = gym.make(env_id)
        env.action_space.seed(seed + idx)
        
        # 1. No-op reset
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        
        # 2. Max and skip 4
        env = MaxAndSkipEnv(env, skip=4)
        
        # 3. Episodic life (for training credit assignment)
        if episodic_life:
            env = EpisodicLifeEnv(env)
            
        # 4. Fire reset if game requires it
        action_meanings = env.unwrapped.get_action_meanings()
        if 'FIRE' in action_meanings and len(action_meanings) >= 3 and action_meanings[1] == 'FIRE':
            env = FireResetEnv(env)
            
        # 5. Warp to 84x84 grayscale
        env = WarpFrame(env, width=84, height=84)
        
        # 6. Clip reward
        if clip_reward:
            env = ClipRewardEnv(env)
            
        # 7. Stack 4 frames -> (4, 84, 84)
        if frame_stack > 1:
            env = FrameStack(env, k=frame_stack)
            
        return env
    return _thunk


def make_vector_atari_envs(
    env_id: str,
    num_envs: int = 16,
    seed: int = 42,
    noop_max: int = 30,
    frame_stack: int = 4,
    clip_reward: bool = True,
    episodic_life: bool = True,
    asynchronous: bool = True
) -> Any:
    """Create vectorized Atari environments (AsyncVectorEnv or SyncVectorEnv)."""
    env_fns = [
        make_atari_env(
            env_id=env_id,
            seed=seed,
            idx=i,
            noop_max=noop_max,
            frame_stack=frame_stack,
            clip_reward=clip_reward,
            episodic_life=episodic_life
        )
        for i in range(num_envs)
    ]
    if asynchronous and num_envs > 1:
        try:
            return gym.vector.AsyncVectorEnv(env_fns)
        except Exception:
            return gym.vector.SyncVectorEnv(env_fns)
    return gym.vector.SyncVectorEnv(env_fns)
