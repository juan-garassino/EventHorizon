"""Regression tests added by the guardian run (see GUARDIAN.md).

Each test reproduces a concrete bug or risky pattern; they run on CPU in a
few seconds and need no data files.
"""
import os
from xml.etree import ElementTree as ET  # parses only our own test output

import numpy as np
import pytest

from eventHorizon.core.isoradial_model import BaseModel
from eventHorizon.math import geodesics as math_geodesics
from eventHorizon.physics import disk_physics
from eventHorizon.physics import geodesics as physics_geodesics
from eventHorizon.physics.fast_geodesics import flux_intrinsic_vec, redshift_vec
from eventHorizon.rendering.export import plotter_export as pe_render
from eventHorizon.visualization import plotter_export as pe_vis

PARAMS_INI = os.path.join(os.path.dirname(__file__), "parameters.ini")


@pytest.mark.parametrize("module", [math_geodesics, physics_geodesics])
def test_fast_root_converges_when_residuals_are_negative(module):
    """G-2: fast_root used `nanmax(f_mid) < tol` (no abs), so it returned
    after ONE bisection whenever every residual was negative -- i.e. a root
    error of up to half a grid cell."""
    def f(x, y):  # root at x = y
        return x - y
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


# --- pass 2: known-answer physics + SVG export -----------------------------

INCL = np.radians(80.0)


def test_redshift_vec_matches_scalar_and_doppler_asymmetry():
    r = np.array([6.0, 10.0, 20.0])
    alpha = np.array([np.pi / 2, 3 * np.pi / 2, 0.3])
    b = np.array([7.0, 9.0, 15.0])
    vec = redshift_vec(r, alpha, INCL, 1.0, b)
    scal = [disk_physics.redshift_factor(*args, INCL, 1.0, bb)
            for args, bb in zip(zip(r, alpha), b)]
    np.testing.assert_allclose(vec, scal, rtol=1e-12)
    # alpha=0, b=0: pure gravitational term 1/sqrt(1-3M/r) = sqrt(2) at ISCO
    assert redshift_vec(np.array([6.0]), np.array([0.0]), INCL, 1.0,
                        np.array([0.0]))[0] == pytest.approx(np.sqrt(2.0))
    # one side of the disk is blueshifted (1+z < 1), the other redshifted
    z = redshift_vec(np.array([10.0, 10.0]), np.array([np.pi / 2, 3 * np.pi / 2]),
                     INCL, 1.0, np.array([10.0, 10.0]))
    assert z.min() < 1.0 < z.max()


def test_flux_intrinsic_zero_inside_isco_and_matches_scalar():
    r = np.array([2.5, 6.0, 7.0, 10.0, 30.0])
    vec = flux_intrinsic_vec(r, 1.0)
    assert vec[0] == 0.0 and vec[1] == 0.0
    scal = [disk_physics.flux_intrinsic(x, 1.0, 1.0) for x in r]
    np.testing.assert_allclose(vec, scal, rtol=1e-12)
    assert vec[3] > vec[2] and vec[3] > vec[4]  # peaks between ISCO and 30M


@pytest.mark.parametrize("mod", [pe_render, pe_vis])
def test_svg_export_is_well_formed(mod, tmp_path):
    """G-4: _write_svg now pretty-prints with ElementTree.indent instead of
    re-parsing through minidom; output must stay valid, single-declaration SVG."""
    out = tmp_path / "x.svg"
    mod.polylines_to_svg([np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])], str(out))
    text = out.read_text()
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert text.count("<?xml") == 1
    root = ET.fromstring(text.split("\n", 1)[1])  # nosec B314 - self-generated
    assert root.tag.endswith("svg")
    assert any(el.tag.endswith(("polyline", "path")) for el in root.iter())
