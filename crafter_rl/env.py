"""Gymnasium adapter for Crafter.

Crafter ships its own old-gym-style API (4-tuple step, custom DiscreteSpace).
Stable-Baselines3 expects the modern Gymnasium API (5-tuple, gymnasium spaces),
so this thin wrapper bridges the two.
"""
import crafter
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CrafterGym(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size=(64, 64), length=10000, seed=None):
        self._env = crafter.Env(size=size, length=length, seed=seed)
        h, w = size
        self.observation_space = spaces.Box(0, 255, (h, w, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(self._env.action_space.n)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.asarray(self._env.reset(), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        obs = np.asarray(obs, dtype=np.uint8)
        # Crafter sets done on both death and time-limit. discount==0 means the
        # player died (true termination); discount==1 at done means truncation.
        discount = info.get("discount", 1.0)
        terminated = bool(done) and discount == 0
        truncated = bool(done) and not terminated
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()
