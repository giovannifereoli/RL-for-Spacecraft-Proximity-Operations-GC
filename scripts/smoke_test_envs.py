#!/usr/bin/env python
"""Minimal smoke runner for formal environment variants.

Runs reset + a few steps for each major environment class.
Does not train, plot, download models, or require a GPU.

Environment modules are loaded via tests/env_loader.py using
Path(__file__).resolve() and importlib (no os.chdir, no edits to research
entrypoints). Historical scripts still rely on directory-local imports.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Headless plotting backend if any dependency touches matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from env_loader import ENV_MODULES, EXPECTED_OBS_SHAPE, make_env  # noqa: E402


def run_one(key: str, n_steps: int = 3) -> None:
    env = make_env(key, tof=15.0, dt=0.5)
    obs = np.asarray(env.reset())
    print(
        f"{key}: reset ok | obs_shape={obs.shape} "
        f"(expected {EXPECTED_OBS_SHAPE[key]}) | finite={bool(np.isfinite(obs).all())}"
    )
    if obs.shape != EXPECTED_OBS_SHAPE[key]:
        raise RuntimeError(f"{key}: unexpected observation shape {obs.shape}")

    for i in range(n_steps):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        step_out = env.step(action)
        if len(step_out) != 4:
            raise RuntimeError(f"{key}: expected Gym 4-tuple step return, got {len(step_out)}")
        obs, reward, done, info = step_out
        obs = np.asarray(obs)
        reward_f = float(reward)
        if not np.isfinite(obs).all() or not np.isfinite(reward_f):
            raise RuntimeError(f"{key}: non-finite obs/reward at step {i}")
        print(
            f"  step {i + 1}: obs_shape={obs.shape} reward={reward_f:.6g} "
            f"done={bool(done)} info_keys={list(info.keys())}"
        )
        if done:
            break


def main() -> int:
    started = time.time()
    failures = []
    for key in sorted(ENV_MODULES):
        try:
            run_one(key)
        except Exception as exc:  # noqa: BLE001 - report all variants, keep full traceback
            failures.append((key, exc))
            print(f"{key}: FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
    elapsed = time.time() - started
    print(f"Completed in {elapsed:.2f}s")
    if failures:
        print(f"{len(failures)} failure(s)", file=sys.stderr)
        return 1
    print("All environment smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
