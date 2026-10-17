"""Regression tests added by the guardian run (see GUARDIAN.md).

Each test reproduces a concrete bug or risky pattern; they run on CPU in a
few seconds and need no data files.
"""
import os

import numpy as np
import pytest

from eventHorizon.core.isoradial_model import BaseModel
from eventHorizon.math import geodesics as math_geodesics
from eventHorizon.physics import geodesics as physics_geodesics

PARAMS_INI = os.path.join(os.path.dirname(__file__), "parameters.ini")


@pytest.mark.parametrize("module", [math_geodesics, physics_geodesics])
def test_fast_root_converges_when_residuals_are_negative(module):
    """G-2: fast_root used `nanmax(f_mid) < tol` (no abs), so it returned
    after ONE bisection whenever every residual was negative -- i.e. a root
    error of up to half a grid cell."""
    f = lambda x, y: x - y  # noqa: E731  root at x = y
    x = np.linspace(0.0, 10.0, 11)
    y = np.array([0.9, 3.8, 7.95])
    roots = module.fast_root(f, x, y, (), tol=1e-6, max_steps=40)
    np.testing.assert_allclose(roots, y, atol=1e-5)


def test_read_parameters_parses_repo_ini():
    params = BaseModel.read_parameters(PARAMS_INI)
    assert params["isoradial_angular_parameters"]["end_angle"] == pytest.approx(np.pi)
    assert params["isoradial_angular_parameters"]["mirror"] is True
    assert params["solver_parameters"]["min_periastron"] == pytest.approx(3.001)
    assert params["plot_params"]["ax_lim"] == (-35, 35)
    assert params["plot_params"]["linestyle"] == "-"
    assert params["plot_params"]["key"] == ""


def test_read_parameters_does_not_execute_code(tmp_path):
    """G-1: read_parameters fed every ini value to eval(), so a config file
    could run arbitrary code. Values must be parsed as literals only."""
    marker = tmp_path / "pwned"
    ini = tmp_path / "evil.ini"
    ini.write_text(f"[s]\nx = open({str(marker)!r}, 'w')\n")
    with pytest.raises(ValueError):
        BaseModel.read_parameters(str(ini))
    assert not marker.exists()
