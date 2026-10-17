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


# --- pass 3: literal parser edge cases --------------------------------------

from eventHorizon.core.isoradial_model import _safe_literal  # noqa: E402


@pytest.mark.parametrize("text, expected", [
    ("3", 3), ("-2.5", -2.5), ("+4", 4), ("1e-8", 1e-8),
    ("np.pi", np.pi), ("pi / 2", np.pi / 2), ("2 * np.pi", 2 * np.pi),
    ("-np.pi", -np.pi), ("2 ** 3", 8), ("1 - 0.5", 0.5), ("np.e", np.e),
    ("(-35, 35)", (-35, 35)), ("[1, 2.0]", [1, 2.0]), ("True", True),
    ("None", None), ('"-"', "-"), ("  7  ", 7),
])
def test_safe_literal_accepts_literals_and_arithmetic(text, expected):
    assert _safe_literal(text) == pytest.approx(expected) if isinstance(
        expected, float) else _safe_literal(text) == expected


@pytest.mark.parametrize("text", [
    "__import__('os')", "__import__('os').system('true')", "os.system",
    "np.load", "np.pi.__class__", "().__class__.__bases__", "open('x')",
    "lambda: 1", "[x for x in (1,)]", "{'a': 1}", "a if b else c",
    "x", "np", "1; 2", "", "1 +", "2 // 1", "1 << 2",
])
def test_safe_literal_rejects_code(text):
    with pytest.raises(ValueError):
        _safe_literal(text)


def test_safe_literal_rejects_huge_power():
    """`9 ** 9 ** 9` would hang the process building a ~370M-digit int."""
    with pytest.raises(ValueError):
        _safe_literal("9 ** 9 ** 9")


# --- pass 3: physics logic bugs ----------------------------------------------

from eventHorizon.math import fast_geodesics as math_fast  # noqa: E402
from eventHorizon.math.spacetime_geometry import SpacetimeGeometry  # noqa: E402
from eventHorizon.physics import fast_geodesics as physics_fast  # noqa: E402
from eventHorizon.physics.spacetime import SpacetimeGeometry as PhysSpacetime  # noqa: E402


@pytest.mark.parametrize("mod", [math_fast, physics_fast])
def test_impact_table_covers_full_circle(mod):
    """G-9: the alpha grid is linspace(0, 2pi, endpoint=False) and the
    interpolator fills NaN outside it, so every particle sampled with
    alpha in (alpha_max, 2pi) got b=NaN and was silently dropped -- a
    missing wedge of the disk image."""
    interp, alpha_grid, _ = mod.build_impact_table(
        1.0, np.radians(80.0), n=0, n_alpha=12, n_r=3, r_min=6.0, r_max=12.0)
    near_2pi = 0.5 * (alpha_grid[-1] + 2 * np.pi)
    b = interp(np.array([[near_2pi, 9.0], [0.0, 9.0], [2 * np.pi - 1e-9, 9.0]]))
    assert np.all(np.isfinite(b))
    assert b[2] == pytest.approx(b[1], rel=1e-6)  # periodic in alpha


@pytest.mark.parametrize("cls", [SpacetimeGeometry, PhysSpacetime])
def test_disk_flux_zero_inside_isco(cls):
    """G-10: calculate_disk_flux only zeroed r <= 3M (comment said ISCO), so
    3M < r < 6M emitted and diverged as r -> 3M via 1/(r-3)."""
    geo = cls(mass=1.0)
    for r in (3.0001, 4.0, 5.9):
        assert geo.calculate_disk_flux(r) == 0.0
    for r in (7.0, 10.0, 30.0):
        assert geo.calculate_disk_flux(r) == pytest.approx(
            float(flux_intrinsic_vec(np.array([r]), 1.0)[0]), rel=1e-12)


# --- pass 3: bare except no longer swallows KeyboardInterrupt ----------------

from eventHorizon.visualization.mode_router import LuminetPointsHandler  # noqa: E402


class _Interrupting:
    """Operand whose arithmetic raises KeyboardInterrupt (simulates Ctrl-C)."""
    def _boom(self, *_):
        raise KeyboardInterrupt
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _boom


@pytest.mark.parametrize("call", [
    lambda h: h._calc_redshift_factor(10.0, 0.5, 1.0, 1.0, _Interrupting()),
    lambda h: h._calc_flux_intrinsic(10.0, _Interrupting(), 1.0),
    lambda h: h._eq13_exact(10.0, _Interrupting(), 0.5, 1.0),
])
def test_mode_router_math_does_not_swallow_keyboard_interrupt(call):
    """G-11: bare `except:` in LuminetPointsHandler math helpers caught
    KeyboardInterrupt/SystemExit, making a long render impossible to stop."""
    handler = LuminetPointsHandler.__new__(LuminetPointsHandler)
    handler.mass = 1.0
    with pytest.raises(KeyboardInterrupt):
        call(handler)


def test_isoradial_table_fallback_is_logged_not_silent(monkeypatch, caplog):
    """G-7: fast-path failures were swallowed with `except Exception: pass`;
    they now fall back the same way but leave a debug log record."""
    import logging

    from eventHorizon.handlers import isoradial

    def boom(*_a, **_k):
        raise RuntimeError("table build failed")

    monkeypatch.setattr(math_fast, "build_impact_table", boom)
    with caplog.at_level(logging.DEBUG, logger=isoradial.__name__):
        assert isoradial._build_table_interpolator(1.0, 1.0, 0, [6.0, 10.0]) is None
    assert any("fallback" in r.getMessage() and r.exc_info for r in caplog.records)
