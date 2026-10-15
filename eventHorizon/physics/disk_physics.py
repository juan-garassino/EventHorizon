"""
Disk physics calculations extracted from LuminetPointsHandler.

Standalone functions for redshift, flux, and impact parameter calculations.
"""
import numpy as np
from scipy.special import ellipj, ellipk, ellipkinc
from typing import Optional


def cos_gamma(alpha: float, incl: float, tol: float = 1e-5) -> float:
    """cos(gamma) = cos(alpha) / sqrt(cos^2(alpha) + cot^2(incl)), Luminet (1979) eq.10."""
    if abs(incl) < tol:
        return 0
    return np.cos(alpha) / np.sqrt(np.cos(alpha) ** 2 + 1 / (np.tan(incl) ** 2))


def redshift_factor(r: float, alpha: float, incl: float, mass: float, b: float) -> float:
    """
    Calculate redshift factor (1+z) using Luminet eq.19.

    (1+z) = (1 + sqrt(M/r^3) * b * sin(incl) * sin(alpha)) / sqrt(1 - 3M/r)

    Scalar twin of ``fast_geodesics.redshift_vec``; the denominator is the
    orbiting-emitter factor 1/u^t, not the static sqrt(1 - 2M/r).
    """
    try:
        if r <= 3.0 * mass:
            return 1.0
        orbital_term = np.sqrt(mass / (r ** 3)) * b * np.sin(incl) * np.sin(alpha)
        grav_term = (1.0 - 3.0 * mass / r) ** (-0.5)
        z = (1.0 + orbital_term) * grav_term
        return z if np.isfinite(z) and z > 0 else 1.0
    except Exception:
        return 1.0


def flux_intrinsic(r: float, accretion_rate: float, mass: float) -> float:
    """
    Novikov-Thorne intrinsic flux, Page & Thorne (1974) eq.15n (Luminet 1979
    eq.15 with the sqrt(3)/2 log factor). Scalar twin of
    ``fast_geodesics.flux_intrinsic_vec``. Returns 0 for r <= 6M (inside ISCO).
    """
    try:
        r_ = r / mass
        if r_ <= 6.0:
            return 0.0
        log_num = (np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))
        log_den = (np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3))
        if log_den <= 0 or log_num <= 0:
            return 0.0
        log_arg = log_num / log_den
        # Page & Thorne (1974) eq.15n: log-term factor is sqrt(3)/2.
        # Luminet (1979) eq.15 has a typo (sqrt(3)/3 == 1/sqrt(3)); use sqrt(3)/2.
        f = (3.0 * mass * accretion_rate / (8.0 * np.pi)) * \
            (1.0 / ((r_ - 3.0) * r ** 2.5)) * \
            (np.sqrt(r_) - np.sqrt(6) + (np.sqrt(3) / 2.0) * np.log(log_arg))
        return max(f, 0.0)
    except Exception:
        return 0.0


def flux_observed(r: float, accretion_rate: float, mass: float, redshift: float) -> float:
    """Observed flux: F_obs = F_intrinsic / (1+z)^4 (Luminet 1979 sec. 4)."""
    try:
        f_intr = flux_intrinsic(r, accretion_rate, mass)
        if redshift <= 0:
            return 0.0
        return f_intr / (redshift ** 4)
    except Exception:
        return 0.0


def calc_q(periastron: float, mass: float) -> float:
    """Compute Q = sqrt((P - 2M)(P + 6M)) (Luminet 1979 eq.4)."""
    return np.sqrt((periastron - 2.0 * mass) * (periastron + 6.0 * mass))


def calc_b_from_periastron(periastron: float, mass: float) -> float:
    """Convert periastron to impact parameter: b = sqrt(P^3 / (P - 2M)) (Luminet 1979 eq.5)."""
    return np.sqrt(periastron ** 3 / (periastron - 2.0 * mass))


def k2(periastron: float, mass: float) -> float:
    """Elliptic modulus squared k^2 = (Q - P + 6M) / (2Q) (Luminet 1979 eq.12)."""
    q = calc_q(periastron, mass)
    return (q - periastron + 6 * mass) / (2 * q)


def zeta_inf(periastron: float, mass: float) -> float:
    """sin^2(zeta_inf) = (Q - P + 2M) / (Q - P + 6M) (Luminet 1979 eq.12)."""
    q = calc_q(periastron, mass)
    arg = (q - periastron + 2 * mass) / (q - periastron + 6 * mass)
    return np.arcsin(np.sqrt(arg))


def eq13(periastron: float, ir_radius: float, ir_angle: float, incl: float,
         n: int, mass: float) -> float:
    """Luminet eq.13 -- returns residual 1 - r * (1/r)(P) (0 at correct periastron).

    1/r = -(Q - P + 2M)/(4MP) + (Q - P + 6M)/(4MP) sn^2(u, k^2), with
    u = gamma/(2 sqrt(P/Q)) + F(zeta_inf, k) for the direct image and
    u = (gamma - 2 n pi)/(2 sqrt(P/Q)) - F(zeta_inf, k) + 2K(k) for order n.
    """
    try:
        z_inf = zeta_inf(periastron, mass)
        q = calc_q(periastron, mass)
        m_ = k2(periastron, mass)
        ell_inf = ellipkinc(z_inf, m_)
        g = np.arccos(cos_gamma(ir_angle, incl))

        if n:  # Ghost image
            ell_k = ellipk(m_)
            ellips_arg = (g - 2.0 * n * np.pi) / (2.0 * np.sqrt(periastron / q)) - ell_inf + 2.0 * ell_k
        else:  # Direct image
            ellips_arg = g / (2.0 * np.sqrt(periastron / q)) + ell_inf

        sn, cn, dn, ph = ellipj(ellips_arg, m_)
        sn2 = sn * sn
        term1 = -(q - periastron + 2.0 * mass) / (4.0 * mass * periastron)
        term2 = ((q - periastron + 6.0 * mass) / (4.0 * mass * periastron)) * sn2

        return 1.0 - ir_radius * (term1 + term2)
    except Exception:
        return float('inf')


def ellipse_fallback(r: float, alpha: float, incl: float) -> float:
    """Newtonian limit of eq.13: b = r sin(gamma) (Luminet 1979 sec. 3)."""
    g = np.arccos(cos_gamma(alpha, incl))
    return r * np.sin(g)


def calc_impact_parameter_exact(r: float, alpha: float, incl: float,
                                n: int, mass: float) -> Optional[float]:
    """
    Calculate impact parameter using exact Luminet eq.13 elliptic integral solver.

    Parameters
    ----------
    r : float
        Disk radius
    alpha : float
        Angle around the disk
    incl : float
        Observer inclination in radians
    n : int
        Image order (0=direct, 1=ghost)
    mass : float
        Black hole mass

    Returns
    -------
    float or None
        Impact parameter, or None if no solution found
    """
    try:
        # For odd-order (ghost) images the observer-frame angle is flipped by pi
        # before solving eq.13 (Luminet 1979; matches solve_for_impact_parameter
        # in the bgmeulem/luminet reference). Without this the ghost impact
        # parameter is the mirror (wrong) value.
        if n % 2 == 1:
            alpha = (alpha + np.pi) % (2 * np.pi)

        min_p = 3.01 * mass
        max_p = max(100.0 * mass, 3.0 * r)

        p_values = np.linspace(min_p, max_p, 50)
        eq13_values = [eq13(p, r, alpha, incl, n, mass) for p in p_values]

        sign_changes = np.where(np.diff(np.sign(eq13_values)))[0]

        if len(sign_changes) == 0:
            # No periastron solution. Following the reference, only order-0
            # photons in the front half of the disk (alpha in front half) are
            # interpolated by the Newtonian ellipse; everything else is absorbed.
            if n == 0 and ((alpha < np.pi / 2) or (alpha > 3 * np.pi / 2)):
                return ellipse_fallback(r, alpha, incl)
            return None

        idx = sign_changes[0]
        p_low = p_values[idx]
        p_high = p_values[idx + 1]

        for _ in range(50):
            p_mid = (p_low + p_high) / 2
            eq13_mid = eq13(p_mid, r, alpha, incl, n, mass)
            eq13_low = eq13(p_low, r, alpha, incl, n, mass)

            if eq13_mid * eq13_low < 0:
                p_high = p_mid
            else:
                p_low = p_mid

            if abs(p_high - p_low) < 1e-6:
                break

        periastron_solution = (p_low + p_high) / 2

        if periastron_solution > 2.0 * mass:
            return calc_b_from_periastron(periastron_solution, mass)
        else:
            return ellipse_fallback(r, alpha, incl)

    except Exception:
        return ellipse_fallback(r, alpha, incl)
