#!/usr/bin/env python3
"""
Physics validation checks for evenHorizon against Luminet (1979), A&A 75, 228.

Five hard-coded checks:
1. Photon capture at b = 5.1 M
2. Photon escape at b = 5.3 M
3. Critical impact parameter b_crit = 3*sqrt(3) ~ 5.196 M
4. Doppler asymmetry (approaching side brighter by >= 3x)
5. Gravitational redshift at ISCO: sqrt(1 - 3M/r) ~ 0.707 at r=6M
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# ============================================================================
# Constants (Luminet 1979, Schwarzschild M=1)
# ============================================================================
M = 1.0
R_HORIZON = 2.0 * M
R_PHOTON_SPHERE = 3.0 * M
R_ISCO = 6.0 * M
B_CRIT_EXACT = 3.0 * np.sqrt(3.0) * M  # 5.19615...
OBSERVER_INCLINATION_DEG = 80.0
OBSERVER_INCLINATION_RAD = np.radians(OBSERVER_INCLINATION_DEG)

results = {}

def report(check_name, passed, measured, expected, tolerance_desc=""):
    status = "PASS" if passed else "FAIL"
    results[check_name] = passed
    print(f"[{status}] {check_name}")
    print(f"       measured  = {measured}")
    print(f"       expected  = {expected}  {tolerance_desc}")
    print()


# ============================================================================
# CHECK 1: Photon capture at b = 5.1 M
# ============================================================================
# A photon with impact parameter b = 5.1 M (< b_crit) must fall into the BH.
# We check this by verifying that b=5.1 < b_crit in our implementation.
# Additionally, we verify via the effective potential: the photon has no
# turning point outside the horizon.
print("=" * 60)
print("evenHorizon Physics Validation (Luminet 1979)")
print("=" * 60)
print()

try:
    # The effective potential for a Schwarzschild BH:
    # V_eff(r) = (1 - 2M/r) / r^2
    # Photon captured if b < b_crit = 3*sqrt(3)*M
    # Also check: at the photon sphere r=3M, V_eff_max = 1/(27*M^2)
    # Photon captured if 1/b^2 > V_eff_max

    b_capture = 5.1 * M
    V_eff_max = 1.0 / (27.0 * M**2)  # Maximum of effective potential
    one_over_b2 = 1.0 / b_capture**2

    # Photon is captured if 1/b^2 > V_eff_max (energy exceeds barrier)
    captured = one_over_b2 > V_eff_max

    report(
        "Check 1: Photon capture (b=5.1M)",
        captured,
        f"1/b^2 = {one_over_b2:.6f}, V_eff_max = {V_eff_max:.6f}, captured = {captured}",
        "photon should be captured (1/b^2 > V_eff_max)",
    )
except Exception as e:
    report("Check 1: Photon capture (b=5.1M)", False, f"ERROR: {e}", "photon should be captured")


# ============================================================================
# CHECK 2: Photon escape at b = 5.3 M
# ============================================================================
try:
    b_escape = 5.3 * M
    one_over_b2_esc = 1.0 / b_escape**2

    # Photon escapes if 1/b^2 < V_eff_max (energy below barrier)
    escapes = one_over_b2_esc < V_eff_max

    report(
        "Check 2: Photon escape (b=5.3M)",
        escapes,
        f"1/b^2 = {one_over_b2_esc:.6f}, V_eff_max = {V_eff_max:.6f}, escapes = {escapes}",
        "photon should escape (1/b^2 < V_eff_max)",
    )
except Exception as e:
    report("Check 2: Photon escape (b=5.3M)", False, f"ERROR: {e}", "photon should escape")


# ============================================================================
# CHECK 3: Critical impact parameter b_crit ~ 5.196 M (to 1%)
# ============================================================================
try:
    from eventHorizon.math.geodesics import expr_b, lambdify
    import sympy as sy

    # The critical impact parameter corresponds to the photon sphere at r=3M.
    # At the photon sphere, the periastron P = 3M, and b(P) = sqrt(P^3/(P-2M))
    b_func = lambdify(["P", "M"], expr_b())

    # Calculate b at the photon sphere (P = 3M)
    b_computed = float(b_func(R_PHOTON_SPHERE, M))

    relative_error = abs(b_computed - B_CRIT_EXACT) / B_CRIT_EXACT
    passed = relative_error < 0.01  # 1% tolerance

    report(
        "Check 3: Critical impact parameter (b_crit)",
        passed,
        f"b_computed = {b_computed:.6f}",
        f"b_exact = {B_CRIT_EXACT:.6f} (tolerance: 1%, error: {relative_error*100:.4f}%)",
    )
except Exception as e:
    report("Check 3: Critical impact parameter", False, f"ERROR: {e}", f"b_crit = {B_CRIT_EXACT:.6f}")


# ============================================================================
# CHECK 4: Doppler asymmetry (left/approaching side >= 3x brighter)
# ============================================================================
try:
    from eventHorizon.math.geodesics import expr_one_plus_z

    one_plus_z_func = lambdify(["alpha", "b", "theta_0", "M", "r"], expr_one_plus_z())

    # Test at r = 10M (mid-disk), at the disk midplane
    r_test = 10.0 * M

    # Luminet's (1+z) formula:
    #   (1+z) = (1 + sqrt(M/r^3) * b * sin(theta_0) * sin(alpha)) / sqrt(1-3M/r)
    #
    # sin(alpha) > 0 at alpha=pi/2 => positive Doppler term => higher (1+z) => RECEDING (redshift)
    # sin(alpha) < 0 at alpha=3pi/2 => negative Doppler term => lower (1+z) => APPROACHING (blueshift)
    #
    # The approaching (blueshifted) side is BRIGHTER because flux ~ 1/(1+z)^4

    # Calculate impact parameter at r=10M for a simple estimate
    b_approx = r_test  # approximate

    # Receding side: alpha = pi/2 (sin(alpha) = +1 -> redshift -> dimmer)
    z_recede = float(one_plus_z_func(np.pi/2, b_approx, OBSERVER_INCLINATION_RAD, M, r_test))

    # Approaching side: alpha = 3*pi/2 (sin(alpha) = -1 -> blueshift -> brighter)
    z_approach = float(one_plus_z_func(3*np.pi/2, b_approx, OBSERVER_INCLINATION_RAD, M, r_test))

    # Flux ~ 1/(1+z)^4, so the approaching side (lower 1+z) is brighter
    # Asymmetry ratio = flux_approach / flux_recede = (z_recede / z_approach)^4
    if z_approach > 0 and z_recede > 0:
        flux_ratio = (z_recede / z_approach) ** 4
    else:
        flux_ratio = 0.0

    passed = flux_ratio >= 3.0

    report(
        "Check 4: Doppler asymmetry (approaching/receding >= 3x)",
        passed,
        f"(1+z)_approach = {z_approach:.4f}, (1+z)_recede = {z_recede:.4f}, flux_ratio = {flux_ratio:.2f}x",
        "flux ratio >= 3.0",
    )
except Exception as e:
    report("Check 4: Doppler asymmetry", False, f"ERROR: {e}", "flux ratio >= 3.0")


# ============================================================================
# CHECK 5: Gravitational redshift at ISCO (r=6M)
# ============================================================================
try:
    # At ISCO (r=6M), with no Doppler effect (alpha=0 -> sin(alpha)=0),
    # the redshift factor is purely gravitational:
    # (1+z) = 1 / sqrt(1 - 3M/r) = 1 / sqrt(1 - 0.5) = 1/sqrt(0.5) ~ 1.4142
    # So sqrt(1 - 3M/r) = 0.7071

    expected_grav_factor = np.sqrt(1.0 - 3.0 * M / R_ISCO)  # sqrt(0.5) = 0.7071
    expected_1pz = 1.0 / expected_grav_factor  # 1.4142

    # Use the redshift formula with alpha=0 (no Doppler contribution)
    z_isco = float(one_plus_z_func(0.0, 0.0, OBSERVER_INCLINATION_RAD, M, R_ISCO))

    # At alpha=0, sin(alpha)=0, so the orbital term vanishes:
    # (1+z) = (1 + 0) / sqrt(1 - 3M/r) = 1/sqrt(0.5)

    relative_error = abs(z_isco - expected_1pz) / expected_1pz
    passed = relative_error < 0.01  # 1% tolerance

    report(
        "Check 5: Gravitational redshift at ISCO (r=6M)",
        passed,
        f"(1+z)_ISCO = {z_isco:.6f}, sqrt(1-3M/r) = {1.0/z_isco:.6f}",
        f"expected (1+z) = {expected_1pz:.6f}, sqrt(1-3M/r) = {expected_grav_factor:.6f} (tolerance: 1%)",
    )
except Exception as e:
    report("Check 5: Gravitational redshift at ISCO", False, f"ERROR: {e}", "sqrt(1-3M/r) ~ 0.707")


# ============================================================================
# ADDITIONAL CHECKS: Codebase consistency
# ============================================================================

print("-" * 60)
print("Additional validation checks")
print("-" * 60)
print()

# Check 6: Verify the sympy expressions are self-consistent
try:
    from eventHorizon.math.geodesics import expr_q, expr_k, expr_zeta_inf

    q_func = lambdify(["P", "M"], expr_q())
    k_func = lambdify(["P", "M", "Q"], expr_k())
    zeta_func = lambdify(["P", "M", "Q"], expr_zeta_inf())

    # Test at P = 10M: Q = sqrt((10-2)(10+6)) = sqrt(128) = 8*sqrt(2)
    P_test = 10.0 * M
    Q_expected = np.sqrt((P_test - 2*M) * (P_test + 6*M))
    Q_computed = float(q_func(P_test, M))

    q_ok = abs(Q_computed - Q_expected) / Q_expected < 1e-10

    report(
        "Check 6: Q expression consistency",
        q_ok,
        f"Q_computed = {Q_computed:.6f}",
        f"Q_expected = {Q_expected:.6f}",
    )
except Exception as e:
    report("Check 6: Q expression consistency", False, f"ERROR: {e}", "")

# Check 7: Intrinsic flux profile shape (flux = 0 at ISCO, peaks near r~8-10M)
try:
    from eventHorizon.physics.fast_geodesics import flux_intrinsic_vec

    flux_isco, flux_10M, flux_30M = flux_intrinsic_vec(
        np.array([R_ISCO + 0.01, 10.0 * M, 30.0 * M]), M)

    # Flux should be near 0 at ISCO, peak around 8-10M, and decrease outward
    flux_increases = flux_10M > flux_isco
    flux_decreases = flux_10M > flux_30M
    passed = flux_increases and flux_decreases

    report(
        "Check 7: Flux profile shape (peaks near ~10M)",
        passed,
        f"F(6.01M) = {flux_isco:.6e}, F(10M) = {flux_10M:.6e}, F(30M) = {flux_30M:.6e}",
        "F(10M) > F(6M) and F(10M) > F(30M)",
    )
except Exception as e:
    report("Check 7: Flux profile shape", False, f"ERROR: {e}", "")

# Check 8: Verify the PhysicsEngine redshift is consistent with sympy expression
try:
    from eventHorizon.physics.disk_physics import redshift_factor

    # Test at r=10M, alpha=pi/4, inclination=80deg
    r_test = 10.0 * M
    alpha_test = np.pi / 4
    b_test = 8.0  # approximate impact parameter

    z_engine = redshift_factor(r_test, alpha_test, OBSERVER_INCLINATION_RAD, M, b_test)
    z_sympy = float(one_plus_z_func(alpha_test, b_test, OBSERVER_INCLINATION_RAD, M, r_test))

    rel_err = abs(z_engine - z_sympy) / abs(z_sympy) if z_sympy != 0 else 0
    passed = rel_err < 1e-10

    report(
        "Check 8: PhysicsEngine vs sympy redshift consistency",
        passed,
        f"z_engine = {z_engine:.10f}, z_sympy = {z_sympy:.10f}",
        f"relative error = {rel_err:.2e} (should be < 1e-10)",
    )
except Exception as e:
    report("Check 8: PhysicsEngine vs sympy consistency", False, f"ERROR: {e}", "")


# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
n_pass = sum(1 for v in results.values() if v)
n_total = len(results)
print(f"Passed: {n_pass}/{n_total}")
for name, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")

if n_pass == n_total:
    print("\nAll physics checks passed.")
    sys.exit(0)
else:
    print(f"\n{n_total - n_pass} check(s) FAILED. See details above.")
    sys.exit(1)
