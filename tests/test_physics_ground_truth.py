"""
Ground-truth physics tests for the Schwarzschild / Luminet (1979) pipeline.

Units: G = c = M = 1 throughout. Every expected value below is derived from
first principles (metric, conserved quantities, closed-form integrals), not
from the code under test.

Run with:  python -m pytest tests/test_physics_ground_truth.py -q
"""
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from eventHorizon.physics.disk_physics import flux_intrinsic, redshift_factor
from eventHorizon.physics.fast_geodesics import (
    flux_intrinsic_vec,
    flux_observed_vec,
    redshift_vec,
)
from eventHorizon.physics.geodesics import (
    expr_b,
    expr_one_plus_z,
    impact_parameter,
    lambda_normalized_bolometric_flux,
    lambda_objective,
    lambdify,
)
from eventHorizon.physics.photon_orbits import (
    critical_impact_parameter,
    deflection_angle,
    integrate_binet,
    is_captured,
    periastron_from_impact,
)

M = 1.0
INCL = np.radians(80.0)  # Luminet's observer: 80 deg from the disk normal
B_CRIT = 3.0 * np.sqrt(3.0)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@pytest.fixture(scope="module")
def objective():
    return lambda_objective()


# ---------------------------------------------------------------------------
# (1) Critical impact parameter b_c = 3 sqrt(3) M
# ---------------------------------------------------------------------------

def test_b_crit_value():
    assert critical_impact_parameter(M) == pytest.approx(5.196152, abs=1e-6)
    # Luminet eq.3, b(P) = sqrt(P^3/(P-2M)), is minimised at the photon sphere P=3M.
    b_of_p = lambdify(["P", "M"], expr_b())
    assert float(b_of_p(3.0, M)) == pytest.approx(B_CRIT, rel=1e-12)
    p = np.linspace(3.01, 50.0, 2000)
    assert np.all(b_of_p(p, M) > B_CRIT)


@pytest.mark.parametrize("b, captured", [(5.1, True), (5.3, False)])
def test_capture_vs_escape(b, captured):
    assert is_captured(b, M) is captured
    # Independent check: integrate the photon orbit equation directly.
    ode_captured, _ = integrate_binet(b, M, dphi=1e-3)
    assert ode_captured is captured


def test_periastron_inverts_impact_parameter():
    b_of_p = lambdify(["P", "M"], expr_b())
    for b in (5.3, 6.0, 10.0, 100.0):
        p = periastron_from_impact(b, M)
        assert p > 3.0
        assert float(b_of_p(p, M)) == pytest.approx(b, rel=1e-10)
    assert np.isnan(periastron_from_impact(5.1, M))


# ---------------------------------------------------------------------------
# (2) Gravitational redshift of the orbiting disk emitter
# ---------------------------------------------------------------------------

def _one_plus_z_first_principles(r, alpha, b, incl):
    """1+z = u^t (1 + Omega * L_photon/E), circular Keplerian emitter.

    u^t = (1 - 2M/r - r^2 Omega^2)^(-1/2) with Omega = sqrt(M/r^3), and the
    photon's conserved L/E projected on the disk axis is b sin(incl) sin(alpha).
    """
    omega = np.sqrt(M / r**3)
    u_t = 1.0 / np.sqrt(1.0 - 2.0 * M / r - r**2 * omega**2)
    return u_t * (1.0 + omega * b * np.sin(incl) * np.sin(alpha))


def test_orbiting_emitter_factor_is_sqrt_1_minus_3_over_r():
    # Pure gravitational + transverse-Doppler factor (sin(alpha) = 0).
    r = np.array([6.0, 10.0, 50.0])
    g = 1.0 / redshift_vec(r, np.zeros_like(r), INCL, M, np.full_like(r, 10.0))
    np.testing.assert_allclose(g, np.sqrt(1.0 - 3.0 / r), rtol=1e-12)
    assert g[0] == pytest.approx(np.sqrt(0.5), rel=1e-12)
    # Not the static-observer factor sqrt(1 - 2M/r): the disk gas orbits.
    assert abs(g[0] - np.sqrt(1.0 - 2.0 / 6.0)) > 0.1


def test_redshift_tends_to_one_far_away():
    r = np.array([1e4, 1e6, 1e8])
    opz = redshift_vec(r, np.full_like(r, np.pi / 2), INCL, M, r)
    np.testing.assert_allclose(opz, 1.0, atol=2e-2)
    assert np.all(np.diff(np.abs(opz - 1.0)) < 0)


def test_redshift_implementations_agree_with_first_principles():
    one_plus_z = lambdify(["alpha", "b", "theta_0", "M", "r"], expr_one_plus_z())
    rng = np.random.default_rng(0)
    r = rng.uniform(6.0, 50.0, 50)
    alpha = rng.uniform(0.0, 2 * np.pi, 50)
    # Physical photons from a radius-r emitter have b of order r or less;
    # (1+z) > 0 then holds. redshift_vec clamps unphysical (1+z) <= 0 to 1.
    b = rng.uniform(0.1, 1.0, 50) * r
    expected = _one_plus_z_first_principles(r, alpha, b, INCL)
    np.testing.assert_allclose(redshift_vec(r, alpha, INCL, M, b), expected, rtol=1e-12)
    np.testing.assert_allclose(one_plus_z(alpha, b, INCL, M, r), expected, rtol=1e-12)
    scalar = [redshift_factor(ri, ai, INCL, M, bi) for ri, ai, bi in zip(r, alpha, b)]
    np.testing.assert_allclose(scalar, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# (3) Doppler asymmetry at ~80 deg inclination
# ---------------------------------------------------------------------------

def test_doppler_asymmetry(objective):
    r = 10.0
    # alpha = 3pi/2 maps to x = b cos(alpha - pi/2) < 0 (left, approaching);
    # alpha = pi/2 maps to x > 0 (right, receding).
    alpha = np.array([3 * np.pi / 2, np.pi / 2])
    b = impact_parameter(alpha, r, INCL, 0, M, objective)
    x = b * np.cos(alpha - np.pi / 2)
    assert x[0] < 0 < x[1]

    g = 1.0 / redshift_vec(np.full(2, r), alpha, INCL, M, b)
    g_approach, g_recede = g
    assert g_approach > 1.0
    assert g_recede < 1.0

    flux = flux_observed_vec(np.full(2, r), M, 1.0 / g)
    assert flux[0] > 3.0 * flux[1]


# ---------------------------------------------------------------------------
# (4) Novikov-Thorne flux
# ---------------------------------------------------------------------------

def _flux_page_thorne(r):
    """Page & Thorne (1974) eq.15n, evaluated as the defining integral.

    F(r) = -Omega_r / (4 pi sqrt(-g) (E - Omega L)^2)
           * int_{r_isco}^{r} (E - Omega L) L_r dr,     sqrt(-g) = r,
    with Schwarzschild circular-orbit E, L, Omega and mdot = 1.
    """
    energy = lambda x: (1 - 2 / x) / np.sqrt(1 - 3 / x)
    ang_mom = lambda x: np.sqrt(x) / np.sqrt(1 - 3 / x)
    omega = lambda x: x**-1.5
    d_ang_mom = lambda x: 0.5 * (x - 6) / (np.sqrt(x) * (1 - 3 / x) ** 1.5 * x)
    integral = quad(lambda x: (energy(x) - omega(x) * ang_mom(x)) * d_ang_mom(x),
                    6.0, r, epsabs=1e-14)[0]
    e_minus_ol = energy(r) - omega(r) * ang_mom(r)
    return 1.5 * r**-2.5 / (4 * np.pi * r * e_minus_ol**2) * integral


def test_flux_zero_at_isco():
    assert flux_intrinsic_vec(np.array([6.0]), M)[0] == 0.0
    assert flux_intrinsic(6.0, 1.0, M) == 0.0
    assert flux_intrinsic_vec(np.array([6.0 + 1e-6]), M)[0] < 1e-10


def test_flux_matches_page_thorne_integral():
    r = np.array([6.5, 7.0, 8.0, 9.5, 12.0, 15.0, 20.0, 40.0])
    expected = np.array([_flux_page_thorne(x) for x in r])
    np.testing.assert_allclose(flux_intrinsic_vec(r, M), expected, rtol=1e-6)
    np.testing.assert_allclose([flux_intrinsic(x, 1.0, M) for x in r], expected, rtol=1e-6)
    # The sympy path is normalised by 3/(8 pi); compare shape at 1+z = 1.
    sym = lambda_normalized_bolometric_flux()(1.0, r, M)
    np.testing.assert_allclose(sym / sym[0], expected / expected[0], rtol=1e-6)


def test_flux_peak_value_quoted_by_luminet():
    """Luminet (1979) below eq.15: (8 pi / 3 M mdot) F_s,max ~ 1.146e-4 at
    r* ~ 9.55. Reproducing it needs the sqrt(3)/2 log factor, not the
    sqrt(3)/3 printed in eq.15 (which would give 1.92e-4)."""
    f_max = flux_intrinsic_vec(np.array([9.55]), M)[0] * 8 * np.pi / 3
    assert f_max == pytest.approx(1.146e-4, rel=1e-3)


def test_flux_peak_and_temperature_ordering():
    res = minimize_scalar(lambda x: -flux_intrinsic_vec(np.array([x]), M)[0],
                          bounds=(6.1, 20.0), method="bounded")
    assert res.x == pytest.approx(9.55, abs=0.1)
    # Local blackbody: T ~ F^(1/4); the hottest annulus is at the flux peak.
    t7, t_peak, t15 = flux_intrinsic_vec(np.array([7.0, res.x, 15.0]), M) ** 0.25
    assert t_peak > t7 and t_peak > t15
    # NOTE: the zero-torque inner boundary suppresses the relativistic flux so
    # strongly near the ISCO that T(7M) < T(15M) (F(7)/F(15) ~ 0.73, confirmed
    # by the Page-Thorne integral above). Only the Newtonian Shakura-Sunyaev
    # profile, T^4 ~ r^-3 (1 - sqrt(6M/r)), gives T(7M) > T(15M).
    assert t7 < t15
    newtonian_t4 = lambda x: x**-3 * (1 - np.sqrt(6 / x))
    assert newtonian_t4(7.0) > newtonian_t4(15.0)


# ---------------------------------------------------------------------------
# (5) Light deflection
# ---------------------------------------------------------------------------

def test_deflection_weak_field_limit():
    b = 100.0
    delta = deflection_angle(b, M)
    assert delta == pytest.approx(4.0 / b, rel=0.04)
    # Weak-field series: 4M/b + 15 pi M^2/(4 b^2) + 128 M^3/(3 b^3) + O(b^-4).
    series = 4.0 / b + 15 * np.pi / (4 * b**2) + 128 / (3 * b**3)
    assert delta == pytest.approx(series, rel=1e-4)


def test_deflection_near_critical_asymptote():
    """Luminet (1979) eq.7: b = b_c + 3.4823 M exp(-mu) as b -> b_c."""
    for b in (5.2, 5.197, 5.1965):
        assert 3.4823 * np.exp(-deflection_angle(b, M)) == pytest.approx(b - B_CRIT, rel=5e-3)


def test_deflection_matches_orbit_integration():
    for b in (5.25, 6.0, 10.0, 100.0):
        _, delta_ode = integrate_binet(b, M, dphi=1e-3)
        assert delta_ode == pytest.approx(deflection_angle(b, M), rel=1e-7)


def test_deflection_grows_sharply_near_b_crit():
    bs = np.array([100.0, 20.0, 10.0, 6.0, 5.3, 5.2, 5.197])
    deltas = np.array([deflection_angle(b, M) for b in bs])
    assert np.all(np.diff(deltas) > 0)
    assert deltas[-2] > 2 * np.pi              # b = 5.2 loops the hole once
    # Log divergence: delta ~ -ln(b - b_c) as b -> b_c.
    assert deltas[-1] > 2 * deltas[-3]
    assert deflection_angle(5.1, M) == np.inf  # captured


# ---------------------------------------------------------------------------
# Disk image solver (Luminet eq.13)
# ---------------------------------------------------------------------------

def test_eq13_periastron_solutions_lie_outside_photon_sphere(objective):
    """Every eq.13 solution must be a real turning point, i.e. b >= b_c."""
    alpha = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    for n in (0, 1):
        for r in (6.0, 10.0, 30.0):
            b = impact_parameter(alpha, r, INCL, n, M, objective)
            assert np.all(np.isfinite(b))
            if n == 1:
                # Ghost photons all wind around the hole.
                assert np.all(b >= B_CRIT - 1e-9)


def test_eq13_solver_accuracy_independent_of_batch_size(objective):
    """A single-angle solve must be as accurate as a batched one.

    Reference: 50-step bisection of Luminet eq.13 in disk_physics.
    """
    from eventHorizon.physics.disk_physics import calc_impact_parameter_exact

    for n in (0, 1):
        for alpha in np.linspace(0.05, 6.2, 12):
            for r in (7.0, 15.0, 30.0):
                b_single = impact_parameter(np.array([alpha]), r, INCL, n, M, objective)[0]
                b_ref = calc_impact_parameter_exact(r, alpha, INCL, n, M)
                assert b_single == pytest.approx(b_ref, abs=1e-4)


def _r_at_sweep(b, sweep):
    """Radius reached by a photon from infinity after sweeping `sweep` rad.

    Integrates the orbit equation u'' = -u + 3Mu^2 (adaptive, tight tolerances)
    independently of the eq.13 elliptic-function solution.
    """
    from scipy.integrate import solve_ivp

    sol = solve_ivp(lambda phi, y: [y[1], -y[0] + 3 * M * y[0] ** 2],
                    [0.0, sweep], [0.0, 1.0 / b], rtol=1e-11, atol=1e-13)
    return 1.0 / sol.y[0, -1]


@pytest.mark.parametrize("r", [10.0, 30.0])
def test_ghost_image_geometry_matches_orbit_integration(r):
    """Ghost photon at plate angle alpha sweeps pi + gamma(alpha) to reach r.

    The photon plane contains the line of sight and the plate direction alpha;
    the direct image meets the disk after sweeping gamma (cos gamma =
    cos alpha / sqrt(cos^2 alpha + cot^2 theta_0), Luminet eq.10), the ghost on
    the next disk crossing, half a turn later.
    """
    from eventHorizon.physics.disk_physics import cos_gamma
    from eventHorizon.physics.geodesics import simulate_flux

    alpha = np.array([0.3, 1.2, 2.5, 3 * np.pi / 2, 5.5])
    alpha_img, b, _, _ = simulate_flux(alpha, r, INCL, 1, M)
    for a, bi in zip(alpha_img, b):
        gamma = np.arccos(cos_gamma(a, INCL))
        # rel 1e-3: near b_c, dr/db is large and fast_root's default 10
        # bisection steps resolve P to ~5e-5 (b to ~1e-5).
        assert _r_at_sweep(bi, np.pi + gamma) == pytest.approx(r, rel=1e-3)
    # Wide part of the ghost is below the shadow (alpha ~ 0 maps to y < 0),
    # the thin part hugs the photon ring above it.
    assert b[0] > b[2] and b[2] < 1.05 * B_CRIT


def test_direct_image_geometry_matches_orbit_integration():
    """Direct photons with a real periastron (b > b_c) sweep gamma(alpha)."""
    from eventHorizon.physics.disk_physics import cos_gamma

    alpha = np.array([2.5, 3 * np.pi / 2])
    for r in (10.0, 30.0):
        b = impact_parameter(alpha, r, INCL, 0, M)
        for a, bi in zip(alpha, b):
            assert bi > B_CRIT
            gamma = np.arccos(cos_gamma(a, INCL))
            assert _r_at_sweep(bi, gamma) == pytest.approx(r, rel=1e-4)


def test_impact_table_has_no_wedge_gap_at_2pi():
    """b(alpha, r) is 2pi-periodic, so the interpolated table must cover the
    whole circle; queries just below 2pi used to return NaN (dropped points)."""
    from eventHorizon.physics.fast_geodesics import impact_table_interpolator

    alpha_grid = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    r_grid = np.array([6.0, 10.0])
    table = 7.0 + np.cos(alpha_grid)[:, None] + r_grid[None, :]
    interp = impact_table_interpolator(alpha_grid, r_grid, table)
    alpha_q = np.array([alpha_grid[-1] + 0.1, 2 * np.pi - 1e-9])
    b = interp(np.column_stack([alpha_q, np.full(2, 8.0)]))
    assert np.all(np.isfinite(b))
    assert b[1] == pytest.approx(table[0, 0] + 2.0, rel=1e-6)
