"""Non-invasive numeric finiteness checks (no physics correctness claims)."""

from __future__ import annotations

import numpy as np
import pytest

from env_loader import ENV_MODULES, make_env


@pytest.mark.parametrize("key", sorted(ENV_MODULES))
def test_state_and_reward_remain_finite(key):
    env = make_env(key, tof=20.0, dt=0.5)
    obs = np.asarray(env.reset())
    assert np.isfinite(obs).all()

    for _ in range(3):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        obs = np.asarray(obs)
        assert np.isfinite(obs).all()
        assert np.isfinite(float(reward))
        assert obs.shape == env.observation_space.shape
        if done:
            break


@pytest.mark.parametrize("key", ["mlp_nominal", "lstm_nominal", "mlp_pert", "lstm_pert"])
def test_mass_component_positive_after_zero_thrust_step(key):
    """Mass is state index 12 in the unscaled IVP state (before MDP appendages)."""
    env = make_env(key, tof=20.0, dt=0.5)
    env.reset()
    env.step(np.zeros(3, dtype=np.float32))
    # After step, env.state is scaled; reverse to physical/adimensional MDP state.
    full = np.asarray(env.scaler_reverse_observation(env.state)).flatten()
    mass = float(full[12])
    assert mass > 0.0
    assert np.isfinite(mass)


@pytest.mark.parametrize("key", sorted(ENV_MODULES))
def test_observation_shape_stable_across_resets(key):
    env = make_env(key, tof=10.0, dt=0.5)
    shapes = []
    for _ in range(3):
        obs = np.asarray(env.reset())
        shapes.append(obs.shape)
        env.step(env.action_space.sample())
    assert len(set(shapes)) == 1
