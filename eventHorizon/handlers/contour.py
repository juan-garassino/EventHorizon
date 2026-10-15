"""
LuminetPointsHandler -- contour-based accretion disk visualization.

This is the main handler for the 'points' / 'luminet' visualization modes.
It generates particle data using exact Luminet physics and renders using
tricontourf for smooth contour plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List

from .base import VisualizationHandler
from .edge_fill import (
    compute_apparent_edge_b,
    add_inner_disk_edge_fill,
    add_apparent_inner_edge_fill,
    add_outer_disk_edge_fill,
)
from ..physics.disk_physics import (
    redshift_factor,
    flux_intrinsic,
    flux_observed,
    calc_impact_parameter_exact,
    cos_gamma,
    eq13,
    calc_q,
    calc_b_from_periastron,
    k2,
    zeta_inf,
    ellipse_fallback,
)
from ..utils.validation import validate_and_suggest


class LuminetPointsHandler(VisualizationHandler):
    """Handler for luminet-style particle visualization (points mode) with enhanced mathematical foundation."""

    def validate_parameters(self) -> bool:
        """Validate parameters for points mode."""
        particle_count = self.params.get('particle_count', 10000)
        power_scale = self.params.get('power_scale', 0.9)

        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if not 0 < power_scale <= 1.0:
            raise ValueError("power_scale must be between 0 and 1")

        return True

    def render(self) -> Tuple[plt.Figure, plt.Axes]:
        """Render luminet-style particle visualization using enhanced sampling approach."""
        validate_and_suggest(
            "draw_blackhole",
            mass=self.mass,
            inclination=self.inclination,
            particle_count=self.params.get('particle_count', 10000),
            power_scale=self.params.get('power_scale', 0.9),
        )

        power_scale = self.params.get('power_scale', 0.9)
        levels = self.params.get('levels', 100)

        return self._render_with_enhanced_sampling(power_scale, levels)

    # ------------------------------------------------------------------
    # Impact parameter calculation (delegates to physics.disk_physics)
    # ------------------------------------------------------------------

    def _calc_original_impact_parameter(self, r: float, alpha: float, incl: float, n: int = 0) -> Optional[float]:
        """Calculate impact parameter using exact Luminet eq.13 elliptic integral solver."""
        return calc_impact_parameter_exact(r, alpha, incl, n, self.mass)

    def _calc_redshift_factor(self, r: float, alpha: float, incl: float, mass: float, b: float) -> float:
        """Calculate redshift factor using Luminet's formula (eq. 19)."""
        return redshift_factor(r, alpha, incl, mass, b)

    def _calc_flux_intrinsic(self, r: float, accretion_rate: float, mass: float) -> float:
        """Calculate intrinsic flux using Novikov-Thorne formula."""
        return flux_intrinsic(r, accretion_rate, mass)

    def _calc_flux_observed(self, r: float, accretion_rate: float, mass: float, redshift_val: float) -> float:
        """Calculate observed flux: F_obs = F_intrinsic / (1+z)^4."""
        return flux_observed(r, accretion_rate, mass, redshift_val)

    # ------------------------------------------------------------------
    # Apparent edge helpers (delegate to edge_fill standalone functions)
    # ------------------------------------------------------------------

    def _compute_apparent_edge_b(self, r_edge: float, angles: np.ndarray) -> np.ndarray:
        """Vectorized computation of apparent edge impact parameters."""
        return compute_apparent_edge_b(r_edge, angles, self.mass, self.inclination)

    def _add_inner_disk_edge_fill(self, ax: plt.Axes, zorder: int = 1) -> None:
        """Black fill for apparent inner disk edge (direct image)."""
        add_inner_disk_edge_fill(ax, self.mass, self.inclination, zorder=zorder)

    def _add_apparent_inner_edge_fill(self, ax: plt.Axes, zorder: int = 1, y_flip: bool = False) -> None:
        """Black fill for apparent inner edge of the black hole (silhouette)."""
        add_apparent_inner_edge_fill(ax, self.mass, self.inclination, zorder=zorder, y_flip=y_flip)

    def _add_outer_disk_edge_fill(self, ax: plt.Axes, zorder: int = 0) -> None:
        """Black fill for apparent outer disk edge (ghost image)."""
        add_outer_disk_edge_fill(ax, self.mass, self.inclination, zorder=zorder)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _plot_direct_image_original(
        self, ax: plt.Axes, points_df: pd.DataFrame,
        levels: int, min_flux: float, max_flux: float, power_scale: float,
    ) -> plt.Axes:
        """Plot direct image: tricontourf + inner disk edge black fill (zorder=1)."""
        points_sorted = points_df.sort_values(by="angle")
        points_filtered = points_sorted

        if points_filtered.empty:
            return ax

        if max_flux > min_flux:
            fluxes = ((points_filtered['flux_o'].abs() + min_flux) / (max_flux + min_flux)).values ** power_scale
        else:
            fluxes = np.ones(len(points_filtered))

        try:
            ax.tricontourf(
                points_filtered['X'].values, points_filtered['Y'].values, fluxes,
                cmap='Greys_r', levels=levels,
                norm=plt.Normalize(0, 1), nchunk=2,
            )
        except Exception:
            ax.scatter(
                points_filtered['X'], points_filtered['Y'],
                c=fluxes, cmap='Greys_r', s=1, alpha=0.7,
            )

        self._add_inner_disk_edge_fill(ax, zorder=1)
        return ax

    def _plot_ghost_image_original(
        self, ax: plt.Axes, points_df: pd.DataFrame,
        levels: int, min_flux: float, max_flux: float, power_scale: float,
    ) -> plt.Axes:
        """Plot ghost image: split inner/outer by apparent edges, different zorders."""
        if points_df.empty:
            return ax

        inner_radius = 6.0 * self.mass
        outer_radius = 50.0 * self.mass

        angles = points_df['angle'].values
        b_vals = points_df['impact_parameter'].values

        unique_angles = np.unique(angles)
        shifted_angles = np.mod(unique_angles + np.pi, 2 * np.pi)

        inner_edge_b = self._compute_apparent_edge_b(inner_radius, shifted_angles)
        outer_edge_b = self._compute_apparent_edge_b(outer_radius, shifted_angles)

        angle_to_inner = dict(zip(unique_angles, inner_edge_b))
        angle_to_outer = dict(zip(unique_angles, outer_edge_b))

        inner_edge_per_particle = np.array([angle_to_inner.get(a, np.nan) for a in angles])
        outer_edge_per_particle = np.array([angle_to_outer.get(a, np.nan) for a in angles])

        mask_inner = b_vals < inner_edge_per_particle
        mask_outer = b_vals > outer_edge_per_particle

        points_inner = points_df[mask_inner]
        points_outer = points_df[mask_outer]

        for i, points_ in enumerate([points_inner, points_outer]):
            if points_.empty:
                continue
            points_ = points_.sort_values(by=['flux_o'], ascending=False)

            if max_flux > min_flux:
                fluxes = ((points_['flux_o'].abs() + min_flux) / (max_flux + min_flux)).values ** power_scale
            else:
                fluxes = np.full(len(points_), 0.5)

            # Ghost inner (i=0) at zorder=1, ghost outer (i=1) at zorder=0
            z = 1 - i
            try:
                ax.tricontourf(
                    points_['X'].values, -points_['Y'].values, fluxes,
                    cmap='Greys_r', norm=plt.Normalize(0, 1), levels=levels,
                    nchunk=2, zorder=z,
                )
            except Exception:
                ax.scatter(
                    points_['X'].values, -points_['Y'].values,
                    c=fluxes, cmap='Greys_r', s=1, alpha=0.4, zorder=z,
                )

        # Ghost black fills
        self._add_apparent_inner_edge_fill(ax, zorder=1, y_flip=False)
        self._add_outer_disk_edge_fill(ax, zorder=0)

        return ax

    # ------------------------------------------------------------------
    # Particle generation
    # ------------------------------------------------------------------

    def _generate_particle_data(self, progress_callback=None) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
        """
        Generate particle data using exact Luminet physics.

        Returns (direct_df, ghost_df, min_flux, max_flux).
        Tries fast vectorized path first, falls back to per-particle bisection.
        """
        show_ghost_image = self.params.get('show_ghost_image', True)
        particle_count = self.params.get('particle_count', 10000)
        inclination_rad = self.inclination * np.pi / 180.0
        accretion_rate = self.params.get('accretion_rate', 1.0)

        # --- Fast path: vectorized lookup table ---
        try:
            from ..physics.fast_geodesics import generate_particles_fast, build_impact_table

            table_direct = None
            table_ghost = None
            try:
                from ..math._fast_geodesics_cy import cy_build_impact_table

                def _cy_interp(mass, incl, n):
                    table, alpha_g, r_g = cy_build_impact_table(
                        mass, incl, n, n_alpha=200, n_r=100,
                    )
                    from ..physics.fast_geodesics import impact_table_interpolator
                    return impact_table_interpolator(alpha_g, r_g, table)

                table_direct = _cy_interp(self.mass, inclination_rad, 0)
                table_ghost = _cy_interp(self.mass, inclination_rad, 1) if show_ghost_image else None
            except ImportError:
                pass

            return generate_particles_fast(
                particle_count, self.mass, inclination_rad,
                accretion_rate=accretion_rate,
                ghost=show_ghost_image,
                table_direct=table_direct,
                table_ghost=table_ghost,
                progress_callback=progress_callback,
            )
        except ImportError:
            pass

        # --- Fallback: per-particle bisection loop ---
        return self._generate_particle_data_legacy(progress_callback)

    def _generate_particle_data_legacy(self, progress_callback=None) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
        """Original per-particle bisection loop (fallback)."""
        show_ghost_image = self.params.get('show_ghost_image', True)
        particle_count = self.params.get('particle_count', 10000)
        inner_radius = 6.0 * self.mass
        outer_radius = 50.0 * self.mass
        inclination_rad = self.inclination * np.pi / 180.0
        accretion_rate = self.params.get('accretion_rate', 1.0)

        direct_particles: List[Dict[str, Any]] = []
        ghost_particles: List[Dict[str, Any]] = []

        for i in range(particle_count):
            if progress_callback and i % 500 == 0:
                progress_callback(i, particle_count)
            elif i % 2000 == 0:
                print(f"   Progress: {i}/{particle_count}")

            r = inner_radius + (outer_radius - inner_radius) * np.random.random()
            theta = np.random.random() * 2 * np.pi

            b_direct = self._calc_original_impact_parameter(r, theta, inclination_rad, n=0)
            b_ghost = self._calc_original_impact_parameter(r, theta, inclination_rad, n=1) if show_ghost_image else None

            if b_direct is not None:
                x_direct = b_direct * np.cos(theta - np.pi / 2)
                y_direct = b_direct * np.sin(theta - np.pi / 2)
                z_factor = self._calc_redshift_factor(r, theta, inclination_rad, self.mass, b_direct)
                flux_direct = self._calc_flux_observed(r, accretion_rate, self.mass, z_factor)
                direct_particles.append({
                    'X': x_direct, 'Y': y_direct, 'flux_o': flux_direct,
                    'radius': r, 'angle': theta, 'impact_parameter': b_direct,
                    'z_factor': z_factor,
                })

            if b_ghost is not None:
                x_ghost = b_ghost * np.cos(theta - np.pi / 2)
                y_ghost = b_ghost * np.sin(theta - np.pi / 2)
                z_ghost = self._calc_redshift_factor(r, theta, inclination_rad, self.mass, b_ghost)
                flux_ghost = self._calc_flux_observed(r, accretion_rate, self.mass, z_ghost)
                ghost_particles.append({
                    'X': x_ghost, 'Y': y_ghost, 'flux_o': flux_ghost,
                    'radius': r, 'angle': theta, 'impact_parameter': b_ghost,
                    'z_factor': z_ghost,
                })

        if progress_callback:
            progress_callback(particle_count, particle_count)

        direct_df = pd.DataFrame(direct_particles)
        ghost_df = pd.DataFrame(ghost_particles) if ghost_particles else pd.DataFrame()

        all_fluxes = list(direct_df['flux_o']) if not direct_df.empty else []
        if not ghost_df.empty:
            all_fluxes.extend(ghost_df['flux_o'])
        max_flux = max(all_fluxes) if all_fluxes else 1.0
        min_flux = 0.0

        return direct_df, ghost_df, min_flux, max_flux

    # ------------------------------------------------------------------
    # Main render pipeline
    # ------------------------------------------------------------------

    def _render_with_enhanced_sampling(self, power_scale: float, levels: int) -> Tuple[plt.Figure, plt.Axes]:
        """Render using Luminet approach with tricontourf for smooth contours."""
        fig, ax = self.setup_figure()
        show_ghost_image = self.params.get('show_ghost_image', True)
        particle_count = self.params.get('particle_count', 10000)

        print(f"Generating {particle_count} particles with exact Luminet physics...")

        direct_df, ghost_df, min_flux, max_flux = self._generate_particle_data(
            progress_callback=self.params.get('progress_callback'),
        )

        print(f"Generated {len(direct_df)} direct and {len(ghost_df)} ghost particles")

        if not direct_df.empty:
            ax = self._plot_direct_image_original(ax, direct_df, levels, min_flux, max_flux, power_scale)
            if show_ghost_image and not ghost_df.empty:
                ax = self._plot_ghost_image_original(ax, ghost_df, levels, min_flux, max_flux, power_scale)

        return fig, ax

    # ------------------------------------------------------------------
    # Scatter rendering support (used by ScatterHandler subclass)
    # ------------------------------------------------------------------

    # Available scatter colormaps
    SCATTER_COLORMAPS = {
        'hot': 'hot',
        'white': None,
        'Greys_r': 'Greys_r',
        'inferno': 'inferno',
        'magma': 'magma',
        'plasma': 'plasma',
        'viridis': 'viridis',
        'cividis': 'cividis',
        'copper': 'copper',
        'bone': 'bone',
    }

    def _render_scatter_particles(
        self, direct_df: pd.DataFrame, ghost_df: pd.DataFrame, power_scale: float,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render particles using scatter plots for authentic Luminet visualization."""
        fig, ax = self.setup_figure()
        show_ghost_image = self.params.get('show_ghost_image', True)
        colormap_name = self.params.get('colormap', 'hot')
        dot_size = self.params.get('dot_size', 2.0)
        use_white = (colormap_name == 'white')
        cmap = None if use_white else self.SCATTER_COLORMAPS.get(colormap_name, colormap_name)

        if direct_df.empty:
            return fig, ax

        # Flux scaling
        all_fluxes: List[float] = []
        if 'flux_o' in direct_df.columns:
            all_fluxes.extend(direct_df['flux_o'].values)
        if not ghost_df.empty and 'flux_o' in ghost_df.columns:
            all_fluxes.extend(ghost_df['flux_o'].values)
        max_flux = max(all_fluxes) if all_fluxes else 1.0
        min_flux = 0.0

        # --- Layer 1 (back): Ghost image ---
        if show_ghost_image and not ghost_df.empty and 'X' in ghost_df.columns:
            visible_ghost = ghost_df[ghost_df['is_visible'] == True] if 'is_visible' in ghost_df.columns else ghost_df
            if not visible_ghost.empty:
                g_flux_norm = (
                    ((visible_ghost['flux_o'].abs() + min_flux) / (max_flux + min_flux)).values ** power_scale
                    if 'flux_o' in visible_ghost.columns else np.full(len(visible_ghost), 0.5)
                )
                g_sizes = dot_size * (0.3 + 1.5 * np.clip(g_flux_norm, 0, 1))

                # NOTE: ghost Y is NOT negated here. The odd-order (ghost) image is
                # already flipped inside the geodesic solver (alpha += pi before
                # solving eq.13, matching the bgmeulem/luminet reference
                # solve_for_impact_parameter). Plotting at -Y would flip a second
                # time and render the ghost upside-down. One flip, at solve time.
                if use_white:
                    ax.scatter(
                        visible_ghost['X'].values, visible_ghost['Y'].values,
                        c='white', s=g_sizes, alpha=g_flux_norm * 0.5,
                        edgecolors='none', zorder=1,
                    )
                else:
                    ax.scatter(
                        visible_ghost['X'].values, visible_ghost['Y'].values,
                        c=g_flux_norm, cmap=cmap, s=g_sizes,
                        alpha=0.6, edgecolors='none', zorder=1,
                    )

        # --- Layer 2 (front): Direct image ---
        if not direct_df.empty and 'X' in direct_df.columns:
            visible = direct_df[direct_df['is_visible'] == True] if 'is_visible' in direct_df.columns else direct_df
            if not visible.empty:
                d_flux_norm = (
                    ((visible['flux_o'].abs() + min_flux) / (max_flux + min_flux)).values ** power_scale
                    if 'flux_o' in visible.columns else np.ones(len(visible))
                )
                d_sizes = dot_size * (0.5 + 2.0 * np.clip(d_flux_norm, 0, 1))

                if use_white:
                    ax.scatter(
                        visible['X'].values, visible['Y'].values,
                        c='white', s=d_sizes, alpha=d_flux_norm * 0.9,
                        edgecolors='none', zorder=2,
                    )
                else:
                    ax.scatter(
                        visible['X'].values, visible['Y'].values,
                        c=d_flux_norm, cmap=cmap, s=d_sizes,
                        alpha=0.9, edgecolors='none', zorder=2,
                    )

        return fig, ax

    # ------------------------------------------------------------------
    # Legacy helpers
    # ------------------------------------------------------------------

    def _particles_to_dataframe(self, particles: List) -> pd.DataFrame:
        """Convert particle objects to DataFrame for visualization."""
        if not particles:
            return pd.DataFrame()

        data = []
        for particle in particles:
            data.append({
                'X': particle.observed_x,
                'Y': particle.observed_y,
                'radius': particle.radius,
                'angle': particle.angle,
                'flux_o': particle.flux,
                'z_factor': particle.redshift_factor,
                'impact_parameter': particle.impact_parameter,
                'brightness': particle.brightness,
                'temperature': particle.temperature,
                'is_visible': particle.is_visible,
            })

        return pd.DataFrame(data)
