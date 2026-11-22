"""Test helper (2026-11-21): src.baselines.tfpp_cam installs stand-ins for `carla` / `shapely` / `agents` and puts carla_garage's team_code
first on sys.path. Inside a training process that is the point; inside a pytest session it leaks: modules that do `try: import carla`
(src/envs/driving_state.py) then take the CARLA path with a stand-in. This fixture factory restores sys.path and drops every module the
test module added that is a stand-in (no __file__) or lives in team_code."""
import os
import sys

import pytest


def isolate_tfpp_imports():
    @pytest.fixture(scope="module", autouse=True)
    def _isolate():
        from src.baselines.tfpp_cam import TEAM_CODE
        before, path = set(sys.modules), list(sys.path)
        yield
        team = os.path.normcase(os.path.abspath(TEAM_CODE))
        for name in set(sys.modules) - before:
            mod = sys.modules.get(name)
            f = getattr(mod, "__file__", None)
            if f is None and name.split(".")[0] in ("carla", "shapely", "agents"):
                del sys.modules[name]
            elif f and os.path.normcase(os.path.abspath(f)).startswith(team):
                del sys.modules[name]
        sys.path[:] = path
    return _isolate
