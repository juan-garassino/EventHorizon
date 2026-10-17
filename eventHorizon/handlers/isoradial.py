"""
IsoradialHandler -- isoradial curve visualization.

Renders isoradial curves (constant-radius contours) for both direct and
ghost images, including the black hole shadow outline.

Uses a precomputed 2D lookup table b(alpha, r) for vectorized interpolation,
with Cython acceleration when available and pure-Python fallback otherwise.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Any, Optional, Tuple, List

from .base import VisualizationHandler
from ..physics.geodesics import UnifiedGeodesics
from ..utils.validation import validate_and_suggest


def _build_table_interpolator(
    mass: float,
    inclination_rad: float,
    image_order: int,
    radii: List[float],
    n_alpha: int = 500,
) -> Optional[RegularGridInterpolator]:
    """Build a RegularGridInterpolator for b(alpha, r).

    Tries Cython cy_build_impact_table first, falls back to pure-Python
    build_impact_table, returns None if both fail.
    """
    r_min = max(min(radii) - 0.1, 3.01 * mass)
    r_max = max(radii) + 0.1
    n_r = max(len(radii) * 4, 50)

    # --- Try Cython ---
    try:
        from ..math._fast_geodesics_cy import cy_build_impact_table
        table, alpha_grid, r_grid = cy_build_impact_table(
            mass, inclination_rad, n=image_order,
            n_alpha=n_alpha, n_r=n_r,
            r_min=r_min, r_max=r_max,
        )
        from ..math.fast_geodesics import _periodic_interp
        return _periodic_interp(alpha_grid, r_grid, table)  # G-9
    except ImportError:
        pass

    # --- Try pure-Python table builder ---
    try:
        from ..math.fast_geodesics import build_impact_table
        interp, _, _ = build_impact_table(
            mass, inclination_rad, n=image_order,
            n_alpha=n_alpha, n_r=n_r,
            r_min=r_min, r_max=r_max,
        )
        return interp
    except Exception:
        logging.getLogger(__name__).debug("fast path failed; using fallback", exc_info=True)

    return None


class IsoradialHandler(VisualizationHandler):
    """Handler for isoradial curve visualization."""

    def validate_parameters(self) -> bool:
        """Validate parameters for isoradials mode."""
        radii = self.params.get('radii')
        if radii is not None and not all(r > 0 for r in radii):
            raise ValueError("All radii must be positive")
        return True

    def get_export_type(self) -> Optional[str]:
        return "isoradials"

    def _compute_isoradial(self, geodesics, radius, inclination_rad,
                           angular_resolution, image_order=0, interp=None):
        """Compute isoradial curve for a given radius and image order.

        If *interp* (a RegularGridInterpolator) is provided, uses vectorized
        table lookup.  Otherwise falls back to per-angle scalar solver.

        Returns (impact_params, valid_angles) or ([], []) on failure.
        """
        angles = np.linspace(0, 2 * np.pi, angular_resolution, endpoint=False)

        # --- Fast path: vectorized table interpolation ---
        if interp is not None:
            pts = np.column_stack([angles, np.full_like(angles, radius)])
            b_arr = interp(pts)
            valid = np.isfinite(b_arr) & (b_arr > 0)
            return b_arr[valid].tolist(), angles[valid].tolist()

        # --- Slow path: per-angle scalar solver ---
        impact_params = []
        valid_angles = []
        for angle in angles:
            try:
                b = geodesics.calculate_impact_parameter(
                    angle, radius, inclination_rad, image_order=image_order,
                )
                if b is not None and b > 0:
                    impact_params.append(b)
                    valid_angles.append(angle)
            except Exception:
                continue

        if len(impact_params) < 5:
            return impact_params, valid_angles

        # Reject outlier impact parameters (spikes from solver noise)
        b_arr = np.array(impact_params)
        median_b = np.median(b_arr)
        mad = np.median(np.abs(b_arr - median_b))
        if mad > 0:
            keep = np.abs(b_arr - median_b) < 6 * mad
            impact_params = list(b_arr[keep])
            valid_angles = [a for a, k in zip(valid_angles, keep) if k]

        return impact_params, valid_angles

    def _plot_isoradial(self, ax, x, y, angular_resolution, **plot_kwargs):
        """Plot an isoradial curve with optional spline smoothing."""
        if len(x) < 4:
            ax.plot(x, y, **plot_kwargs)
            return
        try:
            from scipy.interpolate import splprep, splev
            # Use small smoothing factor to avoid zigzag from exact interpolation
            tck, u = splprep([x, y], s=len(x) * 0.01, per=True, k=3)
            u_new = np.linspace(0, 1, max(angular_resolution, 360))
            x_smooth, y_smooth = splev(u_new, tck)
            ax.plot(x_smooth, y_smooth, **plot_kwargs)
        except Exception:
            ax.plot(x, y, **plot_kwargs)

    def render(self) -> Tuple[plt.Figure, plt.Axes]:
        """Render isoradial curves (direct + ghost) and populate export_data with polylines."""
        validate_and_suggest("plot_isoradials", mass=self.mass, inclination=self.inclination)

        radii = self.params.get('radii')
        if radii is None:
            radii = list(range(6, 51, 5))

        angular_resolution = self.params.get('angular_resolution', 360)
        show_ghost = self.params.get('show_ghost_image', True)

        fig, ax = self.setup_figure()

        geodesics = UnifiedGeodesics(mass=self.mass)
        inclination_rad = self.inclination * np.pi / 180

        # --- Build interpolation tables (Cython → pure-Python → None) ---
        interp_direct = _build_table_interpolator(
            self.mass, inclination_rad, image_order=0, radii=radii,
            n_alpha=max(angular_resolution, 500),
        )
        interp_ghost = None
        if show_ghost:
            interp_ghost = _build_table_interpolator(
                self.mass, inclination_rad, image_order=1, radii=radii,
                n_alpha=max(angular_resolution, 500),
            )

        all_polylines: List[List[Tuple[float, float]]] = []

        # --- Direct isoradials (image_order=0, solid white) ---
        for radius in radii:
            try:
                impact_params, valid_angles = self._compute_isoradial(
                    geodesics, radius, inclination_rad, angular_resolution,
                    image_order=0, interp=interp_direct,
                )
                if len(impact_params) > 3:
                    x = [b * np.cos(a - np.pi / 2) for b, a in zip(impact_params, valid_angles)]
                    y = [b * np.sin(a - np.pi / 2) for b, a in zip(impact_params, valid_angles)]

                    all_polylines.append(list(zip(x, y)))

                    label = f'r = {radius}M' if self.params.get('show_labels', False) else None
                    self._plot_isoradial(
                        ax, x, y, angular_resolution,
                        color='white', alpha=0.7, linewidth=1, label=label,
                    )
            except Exception:
                continue

        # --- Ghost isoradials (image_order=1, dashed) ---
        if show_ghost:
            for radius in radii:
                try:
                    impact_params, valid_angles = self._compute_isoradial(
                        geodesics, radius, inclination_rad, angular_resolution,
                        image_order=1, interp=interp_ghost,
                    )
                    if len(impact_params) > 3:
                        x = [b * np.cos(a - np.pi / 2) for b, a in zip(impact_params, valid_angles)]
                        # No extra Y flip: the n=1 solver already flips alpha by pi,
                        # so negating Y here drew the ghost upside down (GUARDIAN G-14).
                        y = [b * np.sin(a - np.pi / 2) for b, a in zip(impact_params, valid_angles)]

                        all_polylines.append(list(zip(x, y)))

                        ax.plot(x, y, color='white', alpha=0.4, linewidth=0.8, linestyle='--')
                except Exception:
                    continue

        # --- Black hole shadow outline (b_crit = sqrt(27) * M) ---
        b_crit = np.sqrt(27.0) * self.mass
        shadow_angles = np.linspace(0, 2 * np.pi, 200)
        shadow_x = b_crit * np.cos(shadow_angles)
        shadow_y = b_crit * np.sin(shadow_angles)
        ax.plot(shadow_x, shadow_y, color='gray', alpha=0.3, linewidth=0.6, linestyle=':')

        # Store for plotter export
        self.export_data = {"polylines": all_polylines}

        if self.params.get('show_title', True):
            ax.set_title('Isoradial Curves', color='white')

        if self.params.get('show_labels', False):
            ax.legend(loc='upper right')

        return fig, ax
