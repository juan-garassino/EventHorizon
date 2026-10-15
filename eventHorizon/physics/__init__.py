"""Physics calculations for black hole visualization -- single source of truth."""

from .geodesics import (
    UnifiedGeodesics, Geodesics, KerrGeodesics, GeodesicMethod, SpacetimeType,
    impact_parameter, simulate_flux, lambda_objective, lambda_normalized_bolometric_flux,
    fast_root, find_brackets, reorient_alpha, lambdify,
    expr_q, expr_b, expr_r_inv, expr_u, expr_k, expr_zeta_inf, expr_gamma,
    expr_ellipse, expr_fs, expr_r_star, expr_one_plus_z, expr_f0, expr_f0_normalized,
    objective,
)
from .fast_geodesics import (
    generate_particles_fast, build_impact_table,
    redshift_vec, flux_observed_vec, flux_intrinsic_vec, cos_gamma_vec,
    impact_table_interpolator,
)
from .spacetime import SpacetimeGeometry
from .photon_orbits import (
    critical_impact_parameter, periastron_from_impact, is_captured,
    deflection_angle, integrate_binet,
)
from .disk_physics import (
    redshift_factor, flux_intrinsic, flux_observed,
    calc_impact_parameter_exact, cos_gamma, eq13,
    calc_q, calc_b_from_periastron, k2, zeta_inf, ellipse_fallback,
)
