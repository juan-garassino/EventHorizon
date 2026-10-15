"""
Photon orbits in Schwarzschild spacetime: capture, periastron and deflection.

Geometric units G = c = 1. These helpers are independent of the disk
renderer (which solves Luminet's eq.13 directly for each disk point) and are
used to ground-truth the light-bending physics:

* the closed-form impact-parameter / periastron relation (Luminet 1979 eq.5),
* the exact deflection angle mu = 2 phi_inf - pi with phi_inf the complete
  elliptic integral of Luminet eq.6 (Darwin 1959), and
* a direct fixed-step RK4 integration of the photon orbit equation
  (Luminet eq.3 differentiated once) for convergence checks.

References
----------
Luminet, J.-P. (1979), A&A 75, 228.
Darwin, C. (1959), Proc. R. Soc. A 249, 180.
Chandrasekhar, S. (1983), The Mathematical Theory of Black Holes, sec. 20.
"""
import numpy as np
from scipy.special import ellipk, ellipkinc


def critical_impact_parameter(mass: float = 1.0) -> float:
    """Critical impact parameter b_c = 3*sqrt(3)*M.

    Photons with b < b_c are captured; b = b_c asymptotes to the photon
    sphere r = 3M (Luminet 1979, sec. 2: b_c = 3 sqrt(3) M = 5.19695 M).
    """
    return 3.0 * np.sqrt(3.0) * mass


def periastron_from_impact(b: float, mass: float = 1.0) -> float:
    """Periastron P of an unbound photon with impact parameter b.

    Inverts b^2 = P^3 / (P - 2M) (Luminet 1979 eq.5), i.e. the largest real
    root of P^3 - b^2 P + 2 M b^2 = 0. Uses the trigonometric cubic solution:

        P = (2b/sqrt(3)) * cos( (1/3) * arccos(-3*sqrt(3)*M/b) )

    Returns NaN when b < b_c (no turning point: the photon is captured).
    """
    if b < critical_impact_parameter(mass):
        return float("nan")
    return float(2.0 * b / np.sqrt(3.0)
                 * np.cos(np.arccos(-critical_impact_parameter(mass) / b) / 3.0))


def is_captured(b: float, mass: float = 1.0) -> bool:
    """True when a photon from infinity with impact parameter b falls in.

    Equivalent to 1/b^2 exceeding the peak of the photon effective
    potential V(r) = (1 - 2M/r) / r^2, which is 1/(27 M^2) at r = 3M.
    """
    return bool(abs(b) < critical_impact_parameter(mass))


def deflection_angle(b: float, mass: float = 1.0) -> float:
    """Exact light-deflection angle for a photon with impact parameter b.

    With periastron P, Q = sqrt((P - 2M)(P + 6M)), modulus
    k^2 = (Q - P + 6M) / (2Q) and sin^2(zeta_inf) = (Q - P + 2M)/(Q - P + 6M),
    the azimuth swept from infinity to periastron is (Luminet 1979 eq.6)

        phi_inf = 2 sqrt(P/Q) [K(k) - F(zeta_inf, k)],

    and the deflection is delta = 2 phi_inf - pi (Darwin 1959).

    Limits: delta -> 4M/b for b >> M; near b_c, b ~ b_c + 3.4823 M e^-delta
    (Luminet eq.7), i.e. delta diverges logarithmically.
    Returns inf for captured photons.
    """
    p = periastron_from_impact(b, mass)
    if not np.isfinite(p):
        return float("inf")
    q = np.sqrt((p - 2.0 * mass) * (p + 6.0 * mass))
    m = (q - p + 6.0 * mass) / (2.0 * q)
    zeta_inf = np.arcsin(np.sqrt((q - p + 2.0 * mass) / (q - p + 6.0 * mass)))
    return float(4.0 * np.sqrt(p / q) * (ellipk(m) - ellipkinc(zeta_inf, m)) - np.pi)


def _hermite_zero(u0, u1, m0, m1):
    """Root t in [0, 1] of the cubic Hermite interpolant with end values
    u0 > 0 >= u1 and end slopes m0, m1 (already scaled by the step)."""
    def h(t):
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * u0 + (t3 - 2 * t2 + t) * m0
                + (-2 * t3 + 3 * t2) * u1 + (t3 - t2) * m1)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def integrate_binet(b: float, mass: float = 1.0, dphi: float = 1e-3,
                    max_phi: float = 20.0 * np.pi):
    """Integrate a photon from infinity with the orbit equation, fixed-step RK4.

    Uses u = 1/r and the second-order form of Luminet 1979 eq.3,
    (du/dphi)^2 = 2Mu^3 - u^2 + 1/b^2, differentiated w.r.t. phi:

        d^2u/dphi^2 = -u + 3 M u^2,   u(0) = 0,   du/dphi(0) = 1/b.

    Integration stops when the photon crosses the horizon (u >= 1/(2M):
    captured) or returns to infinity (u <= 0 after the turning point:
    escaped). The u = 0 crossing inside the final step is located on the
    cubic Hermite interpolant of (u, du/dphi) at both ends, which keeps the
    escape angle 4th-order accurate like the RK4 steps themselves.

    Returns
    -------
    (captured, deflection) : (bool, float)
        deflection = phi_total - pi for escaping photons, NaN if captured.
    """
    def rhs(y):
        return np.array([y[1], -y[0] + 3.0 * mass * y[0] ** 2])

    y = np.array([0.0, 1.0 / b])
    phi = 0.0
    u_horizon = 1.0 / (2.0 * mass)
    while phi < max_phi:
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dphi * k1)
        k3 = rhs(y + 0.5 * dphi * k2)
        k4 = rhs(y + dphi * k3)
        y_new = y + dphi / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if y_new[0] >= u_horizon:
            return True, float("nan")
        if y_new[0] <= 0.0 and phi > 0.0:
            frac = _hermite_zero(y[0], y_new[0], y[1] * dphi, y_new[1] * dphi)
            return False, float(phi + frac * dphi - np.pi)
        y = y_new
        phi += dphi
    # Still orbiting after max_phi: only possible for b within ~1e-9 of b_c.
    return False, float("inf")
