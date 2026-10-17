"""
Fast vectorized geodesic computations for black hole rendering.

Layer 1: Pure NumPy/SciPy implementation using a precomputed 2D lookup table
for impact parameters b(alpha, r), with vectorized redshift and flux.

Layer 2 (optional): Cython-accelerated table builder — imported transparently
when the compiled extension is available.

Typical speedup vs the per-particle loop in mode_router.py:
  - Layer 1 alone: ~10-20x  (table build ~3-5s, then interpolation is instant)
  - Layer 2 added: ~100-300x (table build ~0.05s)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Vectorized physics (redshift, flux) — trivially parallel over arrays
# ---------------------------------------------------------------------------

def cos_gamma_vec(alpha: np.ndarray, inclination: float) -> np.ndarray:
    """Vectorized cos(gamma) from Luminet eq.13 geometry."""
    cos_a = np.cos(alpha)
    tan_i = np.tan(inclination)
    # Guard against inclination ≈ 0 (face-on)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = cos_a / np.sqrt(cos_a ** 2 + 1.0 / (tan_i ** 2))
    return np.where(np.abs(inclination) < 1e-8, 0.0, result)


def redshift_vec(r: np.ndarray, alpha: np.ndarray,
                 inclination: float, mass: float,
                 b: np.ndarray) -> np.ndarray:
    """
    Vectorized (1+z) — Luminet eq.19.

    (1+z) = (1 + sqrt(M/r^3) * b * sin(incl) * sin(alpha)) / sqrt(1 - 3M/r)
    """
    orbital = np.sqrt(mass / r ** 3) * b * np.sin(inclination) * np.sin(alpha)
    with np.errstate(divide='ignore', invalid='ignore'):
        grav = np.sqrt(1.0 - 3.0 * mass / r)
    z = (1.0 + orbital) / grav
    # Clamp unphysical values (inside photon sphere, etc.)
    return np.where((r > 3.0 * mass) & np.isfinite(z) & (z > 0), z, 1.0)


def flux_intrinsic_vec(r: np.ndarray, mass: float,
                       accretion_rate: float = 1.0) -> np.ndarray:
    """
    Vectorized Novikov-Thorne intrinsic flux with logarithmic correction.
    Returns 0 for r <= 6M (inside ISCO).
    """
    r_ = r / mass
    sqrt_r = np.sqrt(r_)
    sqrt3 = np.sqrt(3.0)
    sqrt6 = np.sqrt(6.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_num = (sqrt_r + sqrt3) * (sqrt6 - sqrt3)
        log_den = (sqrt_r - sqrt3) * (sqrt6 + sqrt3)
        log_term = np.log(np.abs(log_num / log_den))

        # Page & Thorne (1974) eq.15n: log-term factor is sqrt(3)/2.
        # Luminet (1979) eq.15 has a typo (sqrt(3)/3 == 1/sqrt(3)); use sqrt(3)/2.
        f = ((3.0 * mass * accretion_rate) / (8.0 * np.pi) *
             (1.0 / ((r_ - 3.0) * r ** 2.5)) *
             (sqrt_r - sqrt6 + (sqrt3 / 2.0) * log_term))

    return np.where((r_ > 6.0) & np.isfinite(f) & (f > 0), f, 0.0)


def flux_observed_vec(r: np.ndarray, mass: float,
                      redshift: np.ndarray,
                      accretion_rate: float = 1.0) -> np.ndarray:
    """Vectorized F_obs = F_intrinsic / (1+z)^4."""
    f_intr = flux_intrinsic_vec(r, mass, accretion_rate)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = f_intr / redshift ** 4
    return np.where(np.isfinite(result) & (result >= 0), result, 0.0)


# ---------------------------------------------------------------------------
# Lookup-table builder for impact parameter b(alpha, r)
# ---------------------------------------------------------------------------

def _periodic_interp(alpha_grid: np.ndarray, r_grid: np.ndarray,
                     table: np.ndarray) -> RegularGridInterpolator:
    """Interpolator over (alpha, r) that wraps alpha periodically.

    The alpha grid is ``linspace(0, 2pi, endpoint=False)``; without the
    wrap-around column, alpha in (alpha_grid[-1], 2pi) falls outside the grid
    and interpolates to NaN (those particles were dropped).
    """
    alpha_ext = np.append(alpha_grid, 2 * np.pi)
    table_ext = np.vstack([table, table[:1]])
    return RegularGridInterpolator(
        (alpha_ext, r_grid), table_ext,
        method='linear', bounds_error=False, fill_value=np.nan,
    )


def build_impact_table(
    mass: float,
    inclination: float,
    n: int = 0,
    n_alpha: int = 200,
    n_r: int = 100,
    r_min: Optional[float] = None,
    r_max: Optional[float] = None,
    p_grid_size: int = 500,
) -> Tuple[RegularGridInterpolator, np.ndarray, np.ndarray]:
    """
    Build a 2D lookup table b(alpha, r) using the existing vectorized
    root-finder from geodesics.py.

    For each radius in the grid, we call ``impact_parameter()`` with the
    full alpha array — this reuses the battle-tested ``fast_root`` +
    ``lambda_objective`` machinery, so the physics is identical to the
    per-particle solver.

    Parameters
    ----------
    mass, inclination : float
        Black hole mass and observer inclination (radians).
    n : int
        Image order (0 = direct, 1 = ghost).
    n_alpha, n_r : int
        Grid resolution.
    r_min, r_max : float or None
        Disk radii range (defaults to ISCO..50M).
    p_grid_size : int
        Unused; kept for API compatibility. ``impact_parameter`` sizes its
        own periastron scan grid from the radius (GUARDIAN G-13).

    Returns
    -------
    interp : RegularGridInterpolator
        Callable interp((alpha, r)) -> b.
    alpha_grid, r_grid : np.ndarray
        The 1-D axes of the table (for reference / debugging).
    """
    if r_min is None:
        r_min = 6.0 * mass
    if r_max is None:
        r_max = 50.0 * mass

    # --- Try Cython fast path first ---
    try:
        from ._fast_geodesics_cy import cy_build_impact_table
        table, alpha_grid, r_grid = cy_build_impact_table(
            mass, inclination, n=n,
            n_alpha=n_alpha, n_r=n_r,
            r_min=r_min, r_max=r_max,
        )
        return _periodic_interp(alpha_grid, r_grid, table), alpha_grid, r_grid
    except ImportError:
        pass

    # --- Pure-Python fallback ---
    from .geodesics import impact_parameter, lambda_objective

    alpha_grid = np.linspace(0, 2 * np.pi, n_alpha, endpoint=False)
    r_grid = np.linspace(r_min, r_max, n_r)

    obj_func = lambda_objective()

    table = np.empty((n_alpha, n_r), dtype=np.float64)

    for j, r_val in enumerate(r_grid):
        b_arr = impact_parameter(
            alpha_grid, r_val, inclination, n, mass,
            objective_func=obj_func,
        )
        table[:, j] = b_arr

    return _periodic_interp(alpha_grid, r_grid, table), alpha_grid, r_grid


# ---------------------------------------------------------------------------
# Vectorized particle generator (replaces the per-particle Python loop)
# ---------------------------------------------------------------------------

def generate_particles_fast(
    particle_count: int,
    mass: float,
    inclination: float,
    accretion_rate: float = 1.0,
    ghost: bool = True,
    r_inner: Optional[float] = None,
    r_outer: Optional[float] = None,
    table_direct: Optional[RegularGridInterpolator] = None,
    table_ghost: Optional[RegularGridInterpolator] = None,
    progress_callback=None,
    table_kwargs: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    Generate particle data using precomputed lookup tables.

    Drop-in replacement for ``LuminetPointsHandler._generate_particle_data``.
    Returns ``(direct_df, ghost_df, min_flux, max_flux)`` with the same
    column layout: X, Y, flux_o, radius, angle, impact_parameter, z_factor.

    Parameters
    ----------
    particle_count : int
        Number of particles to generate.
    mass : float
        Black hole mass.
    inclination : float
        Observer inclination in **radians**.
    accretion_rate : float
        Disk accretion rate.
    ghost : bool
        Whether to compute ghost (secondary) image particles.
    r_inner, r_outer : float or None
        Disk radii range (defaults to 6M..50M).
    table_direct, table_ghost : RegularGridInterpolator or None
        Precomputed tables. Built on-the-fly if not provided.
    progress_callback : callable or None
        ``callback(current, total)`` for UI progress bars.
    table_kwargs : dict or None
        Extra kwargs for ``build_impact_table`` (grid resolution, etc.).
    """
    if r_inner is None:
        r_inner = 6.0 * mass
    if r_outer is None:
        r_outer = 50.0 * mass

    tkw = table_kwargs or {}

    # --- Step 1: Build or reuse lookup tables ---
    if progress_callback:
        progress_callback(0, particle_count)

    if table_direct is None:
        table_direct, _, _ = build_impact_table(
            mass, inclination, n=0,
            r_min=r_inner, r_max=r_outer, **tkw,
        )
    if ghost and table_ghost is None:
        table_ghost, _, _ = build_impact_table(
            mass, inclination, n=1,
            r_min=r_inner, r_max=r_outer, **tkw,
        )

    if progress_callback:
        progress_callback(int(particle_count * 0.3), particle_count)

    # --- Step 2: Random sampling (vectorized) ---
    rng = np.random.default_rng()
    r = rng.uniform(r_inner, r_outer, size=particle_count)
    theta = rng.uniform(0, 2 * np.pi, size=particle_count)

    # --- Step 3: Impact parameter lookup (vectorized interpolation) ---
    pts = np.column_stack([theta, r])
    b_direct = table_direct(pts)

    if progress_callback:
        progress_callback(int(particle_count * 0.5), particle_count)

    # --- Step 4: Cartesian coordinates ---
    valid_d = np.isfinite(b_direct)
    x_d = np.where(valid_d, b_direct * np.cos(theta - np.pi / 2), np.nan)
    y_d = np.where(valid_d, b_direct * np.sin(theta - np.pi / 2), np.nan)

    # --- Step 5: Redshift & flux (vectorized) ---
    z_d = redshift_vec(r, theta, inclination, mass, b_direct)
    flux_d = flux_observed_vec(r, mass, z_d, accretion_rate)

    if progress_callback:
        progress_callback(int(particle_count * 0.7), particle_count)

    # Build direct DataFrame — drop invalid rows
    direct_df = pd.DataFrame({
        'X': x_d, 'Y': y_d, 'flux_o': flux_d,
        'radius': r, 'angle': theta,
        'impact_parameter': b_direct, 'z_factor': z_d,
    })
    direct_df = direct_df[valid_d].reset_index(drop=True)

    # --- Step 6: Ghost image (same pipeline) ---
    ghost_df = pd.DataFrame()
    if ghost and table_ghost is not None:
        b_ghost = table_ghost(pts)
        valid_g = np.isfinite(b_ghost)
        x_g = np.where(valid_g, b_ghost * np.cos(theta - np.pi / 2), np.nan)
        y_g = np.where(valid_g, b_ghost * np.sin(theta - np.pi / 2), np.nan)
        z_g = redshift_vec(r, theta, inclination, mass, b_ghost)
        flux_g = flux_observed_vec(r, mass, z_g, accretion_rate)
        ghost_df = pd.DataFrame({
            'X': x_g, 'Y': y_g, 'flux_o': flux_g,
            'radius': r, 'angle': theta,
            'impact_parameter': b_ghost, 'z_factor': z_g,
        })
        ghost_df = ghost_df[valid_g].reset_index(drop=True)

    if progress_callback:
        progress_callback(particle_count, particle_count)

    # --- Step 7: Flux range ---
    all_flux = direct_df['flux_o'].tolist()
    if not ghost_df.empty:
        all_flux.extend(ghost_df['flux_o'].tolist())
    max_flux = max(all_flux) if all_flux else 1.0

    return direct_df, ghost_df, 0.0, max_flux
