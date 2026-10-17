"""
Edge fill standalone functions extracted from LuminetPointsHandler.

These functions draw black fills for apparent disk edges (inner, outer,
silhouette) used by the contour and scatter renderers.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def compute_apparent_edge_b(
    r_edge: float,
    angles: np.ndarray,
    mass: float,
    inclination_deg: float,
) -> np.ndarray:
    """
    Vectorized computation of apparent edge impact parameters b(angle)
    for a given radius.  Returns array of b values (NaN where computation fails).

    Parameters
    ----------
    r_edge : float
        Disk radius for the edge.
    angles : np.ndarray
        Array of angles (radians, 0..2pi).
    mass : float
        Black hole mass.
    inclination_deg : float
        Observer inclination in **degrees**.
    """
    inclination_rad = inclination_deg * np.pi / 180.0

    # Try Cython table (fastest)
    try:
        from ..math._fast_geodesics_cy import cy_build_impact_table
        table, alpha_g, r_g = cy_build_impact_table(
            mass, inclination_rad, n=0,
            n_alpha=len(angles), n_r=1,
            r_min=r_edge, r_max=r_edge,
        )
        return table[:, 0]
    except ImportError:
        pass

    # Try vectorized geodesics
    try:
        from ..physics.geodesics import impact_parameter, lambda_objective
        obj_func = lambda_objective()
        return impact_parameter(angles, r_edge, inclination_rad, 0, mass,
                                objective_func=obj_func)
    except Exception:
        logging.getLogger(__name__).debug("fast path failed; using fallback", exc_info=True)

    # Fallback: per-angle solver
    from ..physics.disk_physics import calc_impact_parameter_exact
    b_arr = np.full(len(angles), np.nan)
    for i, angle in enumerate(angles):
        try:
            b = calc_impact_parameter_exact(r_edge, angle, inclination_rad, n=0, mass=mass)
            if b is not None:
                b_arr[i] = b
        except Exception:
            continue
    return b_arr


def add_inner_disk_edge_fill(
    ax: plt.Axes,
    mass: float,
    inclination_deg: float,
    zorder: int = 1,
) -> None:
    """
    Black fill for apparent inner disk edge (direct image).
    Reference: calc_apparent_inner_disk_edge() + fill_between, zorder=1.
    """
    n_angles = 400
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=True)
    b_arr = compute_apparent_edge_b(6.0 * mass, angles, mass, inclination_deg)
    valid = np.isfinite(b_arr)
    if valid.sum() < 3:
        return
    x = b_arr[valid] * np.cos(angles[valid] - np.pi / 2) * 0.99
    y = b_arr[valid] * np.sin(angles[valid] - np.pi / 2) * 0.99
    ax.fill_between(x, y, color='black', zorder=zorder)


def add_apparent_inner_edge_fill(
    ax: plt.Axes,
    mass: float,
    inclination_deg: float,
    zorder: int = 1,
    y_flip: bool = False,
) -> None:
    """
    Black fill for apparent inner edge of the black hole (silhouette).
    This is the black hole silhouette -- min(critical_b, inner_disk_edge_b).
    When y_flip=True, renders in the ghost (negative Y) region.
    """
    n_angles = 400
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=True)
    inner_b = compute_apparent_edge_b(6.0 * mass, angles, mass, inclination_deg)
    critical_b = np.sqrt(27.0) * mass  # 3*sqrt(3) * M for Schwarzschild

    b_arr = np.where(
        (angles > np.pi / 2) & (angles < 3 * np.pi / 2),
        critical_b * 0.99,
        np.minimum(critical_b, np.where(np.isfinite(inner_b), inner_b, critical_b)) * 0.99,
    )

    x = b_arr * np.cos(angles - np.pi / 2)
    y = b_arr * np.sin(angles - np.pi / 2)
    if y_flip:
        y = -y
    ax.fill_between(x, y, color='black', zorder=zorder)


def add_outer_disk_edge_fill(
    ax: plt.Axes,
    mass: float,
    inclination_deg: float,
    zorder: int = 0,
) -> None:
    """
    Black fill for apparent outer disk edge (ghost image).
    Reference: calc_apparent_outer_disk_edge() + fill_between, zorder=0.
    """
    n_angles = 400
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=True)
    b_arr = compute_apparent_edge_b(50.0 * mass, angles, mass, inclination_deg)
    valid = np.isfinite(b_arr)
    if valid.sum() < 3:
        return
    x = b_arr[valid] * np.cos(angles[valid] - np.pi / 2)
    y = b_arr[valid] * np.sin(angles[valid] - np.pi / 2)
    ax.fill_between(x, y, color='black', zorder=zorder)
