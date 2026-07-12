"""Fast smoke tests for formal Gym environment variants (no PPO training)."""

from __future__ import annotations

import numpy as np
import pytest

from env_loader import ENV_MODULES, EXPECTED_OBS_SHAPE, make_env


@pytest.mark.parametrize("key", sorted(ENV_MODULES))
def test_reset_and_step_smoke(key):
    env = make_env(key, tof=15.0, dt=0.5)
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == EXPECTED_OBS_SHAPE[key]

    obs = env.reset()
    obs = np.asarray(obs)
    assert obs.shape == EXPECTED_OBS_SHAPE[key]
    assert np.isfinite(obs).all()

    action = env.action_space.sample()
    assert action.shape == env.action_space.shape

    step_out = env.step(action)
    assert len(step_out) == 4  # old Gym API: obs, reward, done, info
    next_obs, reward, done, info = step_out
    next_obs = np.asarray(next_obs)
    assert next_obs.shape == EXPECTED_OBS_SHAPE[key]
    assert np.isfinite(next_obs).all()
    assert np.isfinite(float(reward))
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


@pytest.mark.parametrize("key", sorted(ENV_MODULES))
def test_zero_action_step(key):
    env = make_env(key, tof=15.0, dt=0.5)
    obs = np.asarray(env.reset())
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_obs, reward, done, info = env.step(action)
    next_obs = np.asarray(next_obs)
    assert next_obs.shape == obs.shape
    assert np.isfinite(next_obs).all()
    assert np.isfinite(float(reward))
