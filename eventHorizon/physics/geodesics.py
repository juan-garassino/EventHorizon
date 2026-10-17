"""
Comprehensive geodesic calculations for black hole spacetime.

This module provides a unified framework for calculating photon trajectories,
impact parameters, and observational effects in curved spacetime around black holes.
Supports both Schwarzschild and Kerr geometries with multiple calculation methods.
"""
import numpy as np
import numpy.typing as npt
import sympy as sy
import scipy.special as sp
from typing import Callable, Optional, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod
from enum import Enum


class SpacetimeType(Enum):
    """Types of spacetime geometries."""
    SCHWARZSCHILD = "schwarzschild"
    KERR = "kerr"
    AUTO = "auto"


class GeodesicMethod(Enum):
    """Available geodesic calculation methods."""
    ANALYTICAL = "analytical"
    NUMERICAL = "numerical"
    HYBRID = "hybrid"
    AUTO = "auto"


# ============================================================================
# MATHEMATICAL EXPRESSIONS FOR SCHWARZSCHILD GEODESICS
# ============================================================================

def expr_q() -> sy.Symbol:
    """Generate a sympy expression for Q."""
    p, m = sy.symbols("P, M")
    return sy.sqrt((p - 2 * m) * (p + 6 * m))


def expr_b() -> sy.Symbol:
    """Generate a sympy expression for b, the impact parameter in the observer's frame.
    
    This represents the impact parameter for photon trajectories in Schwarzschild spacetime.
    """
    p, m = sy.symbols("P, M")
    return sy.sqrt((p**3) / (p - 2 * m))


def expr_r_inv() -> sy.Symbol:
    """Generate a sympy expression for 1/r."""
    p, m, q, u, k = sy.symbols("P, M, Q, u, k")
    sn = sy.Function("sn")
    return (1 / (4 * m * p)) * (-(q - p + 2 * m) + (q - p + 6 * m) * sn(u, k**2) ** 2)


def expr_u() -> sy.Symbol:
    """Generate a sympy expression for the elliptic integral argument in geodesic calculations.
    
    This expression handles both direct and ghost image calculations using elliptic integrals.
    """
    gamma, z_inf, k, p, q, n = sy.symbols("gamma, zeta_inf, k, P, Q, N")
    return sy.Piecewise(
        (gamma / (2 * sy.sqrt(p / q)) + sy.elliptic_f(z_inf, k**2), sy.Eq(n, 0)),
        (
            (gamma - 2 * n * sy.pi) / (2 * sy.sqrt(p / q))
            - sy.elliptic_f(z_inf, k**2)
            + 2 * sy.elliptic_k(k**2),
            True,
        ),
    )


def expr_k() -> sy.Symbol:
    """Generate a sympy expression for the elliptic integral modulus parameter.
    
    The modulus k is used in elliptic integrals for geodesic calculations in curved spacetime.
    """
    p, m, q = sy.symbols("P, M, Q")
    return sy.sqrt((q - p + 6 * m) / (2 * q))


def expr_zeta_inf() -> sy.Symbol:
    """Generate a sympy expression for zeta_inf."""
    p, m, q = sy.symbols("P, M, Q")
    return sy.asin(sy.sqrt((q - p + 2 * m) / (q - p + 6 * m)))


def expr_gamma() -> sy.Symbol:
    """Generate a sympy expression for the geometric angle parameter.
    
    This angle relates the observer's polar angle to the disk inclination angle.
    """
    alpha, theta_0 = sy.symbols("alpha, theta_0")
    return sy.acos(sy.cos(alpha) / sy.sqrt(sy.cos(alpha) ** 2 + sy.tan(theta_0) ** -2))


def expr_ellipse() -> sy.Symbol:
    """Generate a sympy expression for the Newtonian ellipse approximation.
    
    This provides a fallback calculation for cases where relativistic effects are minimal.
    """
    r, alpha, theta_0 = sy.symbols("r, alpha, theta_0")
    return r / sy.sqrt(1 + (sy.tan(theta_0) ** 2) * (sy.cos(alpha) ** 2))


def expr_fs() -> sy.Symbol:
    """Generate an expression for the intrinsic flux of an accretion disk.

    This represents the standard Shakura-Sunyaev disk model flux profile.
    """
    m, rstar, mdot = sy.symbols(r"M, r^*, \dot{m}")
    return (
        ((3 * m * mdot) / (8 * sy.pi))
        * (1 / ((rstar - 3) * rstar ** (5 / 2)))
        * (
            sy.sqrt(rstar)
            - sy.sqrt(6)
            # Page & Thorne (1974) eq.15n: the log-term factor is sqrt(3)/2.
            # Luminet (1979) eq.15 has a typo (sqrt(3)/3); use the correct value.
            + (sy.sqrt(3) / 2)
            * sy.ln(
                ((sy.sqrt(rstar) + sy.sqrt(3)) * (sy.sqrt(6) - sy.sqrt(3)))
                / ((sy.sqrt(rstar) - sy.sqrt(3)) * (sy.sqrt(6) + sy.sqrt(3)))
            )
        )
    )


def expr_r_star() -> sy.Symbol:
    """Generate an expression for r^*, the radial coordinate normalized by the black hole mass."""
    m, r = sy.symbols("M, r")
    return r / m


def expr_one_plus_z() -> sy.Symbol:
    """Generate an expression for the total redshift factor (1+z).
    
    This includes both gravitational redshift and Doppler shift from orbital motion.
    """
    m, r, theta_0, alpha, b = sy.symbols("M, r, theta_0, alpha, b")
    return (1 + sy.sqrt(m / r**3) * b * sy.sin(theta_0) * sy.sin(alpha)) / sy.sqrt(
        1 - 3 * m / r
    )


def expr_f0() -> sy.Symbol:
    """Generate an expression for the observed bolometric flux."""
    fs, opz = sy.symbols("F_s, z_op")
    return fs / opz**4


def expr_f0_normalized() -> sy.Symbol:
    """Generate an expression for the normalized observed bolometric flux."""
    m, mdot = sy.symbols(r"M, \dot{m}")
    return expr_f0() / ((8 * sy.pi) / (3 * m * mdot))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def lambdify(*args, **kwargs) -> Callable:
    """Enhanced lambdify function with support for elliptic functions.
    
    This extends sympy's lambdify to handle Jacobi elliptic functions and elliptic integrals
    commonly used in geodesic calculations.
    """
    kwargs["modules"] = kwargs.get(
        "modules",
        [
            "numpy",
            {
                "sn": lambda u, m: sp.ellipj(u, m)[0],
                "elliptic_f": lambda phi, m: sp.ellipkinc(phi, m),
                "elliptic_k": lambda m: sp.ellipk(m),
            },
        ],
    )
    return sy.lambdify(*args, **kwargs)


def objective() -> sy.Symbol:
    """Generate a sympy expression for the objective function."""
    r = sy.symbols("r")
    return 1 - r * expr_r_inv()


def lambda_objective() -> Callable[[float, float, float, float, int, float], float]:
    """Generate a lambdified objective function."""
    s = (
        objective()
        .subs({"u": expr_u()})
        .subs({"zeta_inf": expr_zeta_inf()})
        .subs({"gamma": expr_gamma()})
        .subs({"k": expr_k()})
        .subs({"Q": expr_q()})
    )
    return lambdify(("P", "alpha", "theta_0", "r", "N", "M"), s)


def lambda_normalized_bolometric_flux() -> Callable[[float, float, float], float]:
    """Generate the normalized bolometric flux function."""
    return sy.lambdify(
        ("z_op", "r", "M"),
        (
            expr_f0()
            .subs({"F_s": expr_fs()})
            .subs({"M": 1, r"\dot{m}": 1})
            .subs({"r^*": expr_r_star()})
        )
        / (3 / (8 * sy.pi)),
    )


# ============================================================================
# NUMERICAL ALGORITHMS
# ============================================================================

def fast_root(
    f: Callable,
    x: npt.NDArray[float],
    y: npt.NDArray[float],
    args: Tuple[Any, ...],
    tol: float = 1e-6,
    max_steps: int = 10,
) -> npt.NDArray[float]:
    """Efficient root finding algorithm for vectorized geodesic calculations.
    
    Uses bracketing and bisection methods optimized for finding periastron distances
    in black hole geodesic problems.
    """
    xmin, xmax = find_brackets(f, x, y, args, tol, max_steps)
    if np.all(np.isnan(xmin)):
        # No sign change anywhere: nothing to bisect (and nanmax below would
        # warn "All-NaN slice"). Callers fall back to the ellipse on NaN.
        return xmin

    sign_min = np.sign(f(xmin, y, *args))
    sign_max = np.sign(f(xmax, y, *args))

    for i in range(max_steps):
        xmid = 0.5 * (xmin + xmax)
        f_mid = f(xmid, y, *args)
        sign_mid = np.sign(f_mid)

        if np.nanmax(np.abs(f_mid)) < tol:
            return xmid
        else:
            xmin = np.where(sign_min == sign_mid, xmid, xmin)
            xmax = np.where(sign_max == sign_mid, xmid, xmax)

    return xmid


def find_brackets(
    f: Callable,
    x: npt.NDArray[float],
    y: npt.NDArray[float],
    args: Tuple[Any, ...],
    tol: float = 1e-6,
    max_steps: int = 10,
) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Find bracketing intervals for root finding in geodesic calculations.
    
    Identifies sign changes in the objective function to bracket roots efficiently.
    """
    xx, yy = np.meshgrid(x, y)
    objective = f(xx, yy, *args)
    flip_mask = (np.diff(np.sign(objective), axis=1) != 0) & (
        ~np.isnan(objective[:, :-1])
    )
    i_at_sign_flip = np.where(
        np.any(flip_mask, axis=1), np.argmax(flip_mask, axis=1), -1
    )

    xmin = np.where(i_at_sign_flip != -1, x[i_at_sign_flip], np.nan)
    xmax = np.where(i_at_sign_flip != -1, x[i_at_sign_flip + 1], np.nan)
    return xmin, xmax


def reorient_alpha(alpha: Union[float, npt.NDArray[float]], n: int) -> Union[float, npt.NDArray[float]]:
    """Reorient polar angles for higher-order images.
    
    Adjusts coordinate angles for secondary (ghost) images that appear at different
    angular positions due to light ray deflection around the black hole.
    """
    return np.where(np.asarray(n) > 0, (alpha + np.pi) % (2 * np.pi), alpha)


def impact_parameter(
    alpha: npt.NDArray[np.float64],
    r_value: float,
    theta_0: float,
    n: int,
    m: float,
    objective_func: Optional[Callable] = None,
    **root_kwargs
) -> npt.NDArray[np.float64]:
    """Calculate impact parameters for photon trajectories in curved spacetime.
    
    Computes the impact parameter b for each viewing angle, accounting for
    relativistic light bending effects around the black hole.
    """
    if objective_func is None:
        objective_func = lambda_objective()

    ellipse = lambdify(["r", "alpha", "theta_0"], expr_ellipse())
    b = lambdify(["P", "M"], expr_b())

    # For odd-order (ghost) images the observer-frame angle is flipped by pi
    # before solving eq.13 (Luminet 1979; see solve_for_impact_parameter in the
    # bgmeulem/luminet reference). Without this flip the ghost periastron solved
    # here is wrong (mirror of the correct value).
    alpha_solve = (np.asarray(alpha) + np.pi) % (2 * np.pi) if (n % 2 == 1) else alpha

    # Periastron scan grid. A photon reaching the disk at radius r has
    # periastron P <= r, so the grid must extend to (at least) r; a fixed
    # upper bound of 50 left large radii unbracketed and silently fell back
    # to the Newtonian ellipse (GUARDIAN G-13).
    p_max = max(50.0, 1.05 * float(r_value))
    p_arr = fast_root(
        objective_func,
        np.linspace(2.1, p_max, 1000),
        alpha_solve,
        (theta_0, r_value, n, m),
        **root_kwargs
    )
    return np.where(np.isnan(p_arr), ellipse(r_value, alpha_solve, theta_0), b(p_arr, m))


def simulate_flux(
    alpha: npt.NDArray[np.float64],
    r: float,
    theta_0: float,
    n: int,
    m: float,
    objective_func: Optional[Callable] = None,
    root_kwargs: Optional[Dict[Any, Any]] = None,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Complete flux simulation including geodesic effects and redshift.
    
    Performs end-to-end calculation from impact parameters to observed flux,
    including all relativistic corrections for black hole accretion disk imaging.
    """
    flux = lambda_normalized_bolometric_flux()
    one_plus_z = sy.lambdify(["alpha", "b", "theta_0", "M", "r"], expr_one_plus_z())
    root_kwargs = root_kwargs if root_kwargs else {}

    b = impact_parameter(alpha, r, theta_0, n, m, objective_func, **root_kwargs)
    opz = one_plus_z(alpha, b, theta_0, m, r)

    return reorient_alpha(alpha, n), b, opz, flux(opz, r, m)


# ============================================================================
# UNIFIED GEODESICS CLASS
# ============================================================================

class UnifiedGeodesics:
    """Unified geodesic calculations for all spacetime types."""
    
    def __init__(
        self, 
        mass: float = 1.0, 
        spin: float = 0.0,
        spacetime: SpacetimeType = SpacetimeType.AUTO,
        method: GeodesicMethod = GeodesicMethod.AUTO
    ):
        """Initialize unified geodesics calculator.
        
        Parameters
        ----------
        mass : float
            Black hole mass
        spin : float
            Dimensionless spin parameter (0 <= a <= 1)
        spacetime : SpacetimeType
            Type of spacetime geometry
        method : GeodesicMethod
            Calculation method
        """
        self.mass = mass
        self.spin = spin
        self.spacetime = spacetime
        self.method = method
        
        # Auto-select spacetime type
        if spacetime == SpacetimeType.AUTO:
            self.spacetime = SpacetimeType.KERR if abs(spin) > 1e-10 else SpacetimeType.SCHWARZSCHILD
        
        # Method selection criteria
        self._method_thresholds = {
            'radius_threshold': 10.0 * mass,
            'precision_threshold': 1e-6,
            'vectorization_threshold': 100
        }
        
        # Initialize lambdified functions for performance
        self._init_lambdified_functions()
    
    def _init_lambdified_functions(self):
        """Initialize commonly used lambdified functions."""
        self.ellipse_func = lambdify(["r", "alpha", "theta_0"], expr_ellipse())
        self.b_func = lambdify(["P", "M"], expr_b())
        self.objective_func = lambda_objective()
        self.one_plus_z_func = lambdify(["alpha", "b", "theta_0", "M", "r"], expr_one_plus_z())
        self.normalized_flux_func = lambda_normalized_bolometric_flux()
    
    def calculate_impact_parameter(
        self,
        alpha: Union[float, npt.NDArray[np.float64]],
        radius: float,
        inclination: float,
        image_order: int = 0,
        **kwargs
    ) -> Union[float, npt.NDArray[np.float64]]:
        """Calculate impact parameter using the best available method."""
        alpha_array = np.atleast_1d(alpha)
        
        # Select calculation method based on spacetime type and method preference
        if self.method == GeodesicMethod.ANALYTICAL or (
            self.method == GeodesicMethod.AUTO and radius > self._method_thresholds['radius_threshold']
        ):
            # Use analytical methods for distant regions
            result = impact_parameter(
                alpha_array, radius, inclination, image_order, self.mass, 
                objective_func=self.objective_func, **kwargs
            )
        elif self.method == GeodesicMethod.NUMERICAL:
            # Use numerical methods for high precision
            result = impact_parameter(
                alpha_array, radius, inclination, image_order, self.mass, 
                objective_func=self.objective_func, **kwargs
            )
        else:  # HYBRID or AUTO
            # Use hybrid approach combining analytical and numerical methods
            result = impact_parameter(
                alpha_array, radius, inclination, image_order, self.mass, 
                objective_func=self.objective_func, **kwargs
            )
        
        return float(result[0]) if np.isscalar(alpha) else result
    
    def calculate_periastron(
        self,
        alpha: float,
        radius: float,
        inclination: float,
        image_order: int = 0,
        **kwargs
    ) -> Optional[float]:
        """Calculate periastron distance."""
        try:
            p_arr = fast_root(
                self.objective_func,
                np.linspace(2.1, max(50.0, 1.05 * float(radius)), 1000),  # G-13
                np.array([alpha]),
                (inclination, radius, image_order, self.mass),
                **kwargs
            )
            return float(p_arr[0]) if not np.isnan(p_arr[0]) else None
        except Exception:
            return None
    
    def calculate_redshift_factor(
        self, 
        radius: float, 
        angle: float, 
        inclination: float, 
        impact_parameter: float
    ) -> float:
        """Calculate gravitational redshift factor (1+z)."""
        return float(self.one_plus_z_func(angle, impact_parameter, inclination, self.mass, radius))
    
    def calculate_observed_flux(
        self, 
        radius: float, 
        redshift_factor: float
    ) -> float:
        """Calculate observed flux including redshift effects."""
        return float(self.normalized_flux_func(redshift_factor, radius, self.mass))
    
    def simulate_complete_flux(
        self,
        alpha: npt.NDArray[np.float64],
        radius: float,
        inclination: float,
        image_order: int = 0,
        **kwargs
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Simulate complete flux calculation including impact parameters."""
        return simulate_flux(alpha, radius, inclination, image_order, self.mass, **kwargs)
    
    def get_ellipse_fallback(
        self, 
        radius: float, 
        alpha: Union[float, npt.NDArray[np.float64]], 
        theta_0: float
    ) -> Union[float, npt.NDArray[np.float64]]:
        """Get ellipse approximation as fallback."""
        return self.ellipse_func(radius, alpha, theta_0)
    
    def reorient_coordinates(
        self, 
        alpha: Union[float, npt.NDArray[np.float64]], 
        image_order: int
    ) -> Union[float, npt.NDArray[np.float64]]:
        """Reorient polar angle coordinates for ghost images."""
        return reorient_alpha(alpha, image_order)
    
    def calculate_orbital_velocity(self, radius: float) -> float:
        """Calculate orbital velocity at given radius."""
        if self.spacetime == SpacetimeType.SCHWARZSCHILD:
            if radius <= 3.0 * self.mass:  # Inside ISCO
                return np.sqrt(self.mass / (3.0 * self.mass))
            return np.sqrt(self.mass / radius)
        else:  # Kerr spacetime
            # For Kerr spacetime, ISCO depends on spin - simplified for now
            isco_radius = 3.0 * self.mass  # Would be spin-dependent in full implementation
            if radius <= isco_radius:
                return np.sqrt(self.mass / isco_radius)
            return np.sqrt(self.mass / radius)
    
    def calculate_light_bending_angle(self, impact_parameter: float) -> float:
        """Calculate light bending angle."""
        if impact_parameter <= 0:
            return np.pi
        return 4.0 * self.mass / impact_parameter  # Weak field approximation
    
    def calculate_time_dilation(self, radius: float) -> float:
        """Calculate gravitational time dilation factor."""
        if radius <= 2.0 * self.mass:
            return 0.0
        return np.sqrt(1.0 - 2.0 * self.mass / radius)
    
    def get_method_info(self) -> Dict[str, Any]:
        """Get information about available methods and current settings."""
        return {
            'spacetime': self.spacetime.value,
            'method': self.method.value,
            'mass': self.mass,
            'spin': self.spin,
            'thresholds': self._method_thresholds.copy(),
            'available_methods': [method.value for method in GeodesicMethod],
            'available_spacetimes': [spacetime.value for spacetime in SpacetimeType],
            'mathematical_expressions': [
                'Impact parameter calculations',
                'Elliptic integral geodesics', 
                'Redshift and flux calculations',
                'Newtonian fallback approximations',
                'Root finding algorithms'
            ]
        }


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# For backward compatibility with existing code
Geodesics = UnifiedGeodesics
KerrGeodesics = UnifiedGeodesics  # Can be specialized later