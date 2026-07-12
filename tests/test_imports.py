"""Importability checks for formal environment modules."""

from __future__ import annotations

from pathlib import Path

import pytest

from env_loader import ENV_MODULES, load_env_module


@pytest.mark.parametrize("key", sorted(ENV_MODULES))
def test_environment_modules_import(key):
    module = load_env_module(key)
    assert hasattr(module, "ArpodCrtbp")
    # Ensure we load the real tracked Environment*.py, not a stub.
    expected = ENV_MODULES[key].resolve()
    assert Path(module.__file__).resolve() == expected
