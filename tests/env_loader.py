"""Shared helpers for loading experiment Environment modules without package installs.

Historical training scripts use directory-local imports (e.g. from Environment import ...).
Tests and smoke load Environment*.py via importlib and Path(__file__).resolve() instead of
modifying those research entrypoints or calling os.chdir().
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ENV_MODULES = {
    "mlp_nominal": ROOT / "MLP" / "Environment.py",
    "lstm_nominal": ROOT / "LSTM" / "Environment.py",
    "mlp_pert": ROOT / "MLP" / "EnvironmentPert.py",
    "lstm_pert": ROOT / "LSTM" / "EnvironmentPert.py",
    "mlp_constang": ROOT / "MLPconstAng" / "Environment.py",
    "lstm_constang": ROOT / "LSTMconstAng" / "Environment.py",
}

EXPECTED_OBS_SHAPE = {
    "mlp_nominal": (16,),
    "lstm_nominal": (16,),
    "mlp_pert": (16,),
    "lstm_pert": (16,),
    "mlp_constang": (18,),
    "lstm_constang": (18,),
}


def load_env_module(key: str) -> ModuleType:
    path = ENV_MODULES[key]
    mod_name = f"spo_env_{key}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load environment module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def default_kwargs(tof: float = 20.0, dt: float = 0.5) -> dict:
    """Constructor kwargs aligned with formal training scripts (short ToF for tests)."""
    m_star = 6.0458e24
    l_star = 3.844e8
    t_star = 375200.0
    mass = 21000.0
    x0t_state = np.array(
        [
            1.02206694e00,
            -1.32282592e-07,
            -1.82100000e-01,
            -1.69229909e-07,
            -1.03353155e-01,
            6.44013821e-07,
        ]
    )
    x0r_state = np.array(
        [
            1.08357767e-13,
            1.32282592e-07,
            -4.12142542e-13,
            1.69229909e-07,
            -3.65860120e-13,
            -6.44013821e-07,
        ]
    )
    x0r_mass = np.array([mass / m_star])
    x0_time_rem = np.array([tof / t_star])
    x0ivp_vec = np.concatenate((x0t_state, x0r_state, x0r_mass, x0_time_rem))
    x0ivp_std_vec = np.absolute(
        np.concatenate(
            (
                np.zeros(6),
                5.0 * np.ones(3) / l_star,
                0.5 * np.ones(3) / (l_star / t_star),
                0.005 * x0r_mass,
                np.zeros(1),
            )
        )
    )
    return dict(
        max_time=tof,
        dt=dt,
        rho_max=70,
        rhodot_max=6,
        x0ivp=x0ivp_vec,
        x0ivp_std=x0ivp_std_vec,
        ang_corr=np.deg2rad(20),
        safety_radius=1,
        safety_vel=0.01,
    )


def make_env(key: str, tof: float = 20.0, dt: float = 0.5, **overrides):
    module = load_env_module(key)
    kwargs = default_kwargs(tof=tof, dt=dt)
    kwargs.update(overrides)
    return module.ArpodCrtbp(**kwargs)
