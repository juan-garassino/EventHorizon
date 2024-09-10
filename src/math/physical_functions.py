import numpy as np
import sympy as sp
import scipy.special as scp
import warnings
from .numerical_functions import NumericalFunctions
from .geometric_functions import GeometricFunctions

class PhysicalFunctions:
    @staticmethod
    def calculate_luminet_equation_13(periastron: float, emission_radius: float, emission_angle: float, black_hole_mass: float, inclination: float, image_order: int = 0, tolerance: float = 1e-6, verbose: bool = False) -> float:
        """Relation between radius (where photon was emitted in accretion disk), argument and P."""
        if verbose:
            print(f"calculate_luminet_equation_13 input: periastron: {periastron}, emission_radius: {emission_radius}, emission_angle: {emission_angle}, black_hole_mass: {black_hole_mass}, inclination: {inclination}, image_order: {image_order}")

        zeta_infinity = NumericalFunctions.calculate_zeta_infinity(periastron, black_hole_mass, verbose=verbose)
        q_parameter = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose=verbose)
        squared_modulus = NumericalFunctions.calculate_squared_elliptic_integral_modulus(periastron, black_hole_mass, verbose=verbose)

        # Check for invalid q_parameter
        if q_parameter <= 0:
            if verbose:
                print(f"Invalid q_parameter value: {q_parameter}")
            return np.nan

        # Use elliptic_f instead of ellipkinc
        elliptic_integral_infinity = scp.ellipkinc(zeta_infinity, squared_modulus)
        gamma = np.arccos(GeometricFunctions.calculate_cos_gamma(emission_angle, inclination))

        # Check for division by zero or invalid values
        sqrt_term = np.sqrt(periastron / q_parameter)
        if sqrt_term == 0:
            if verbose:
                print(f"Invalid sqrt_term value: {sqrt_term}")
            return np.nan

        if image_order:  # higher order image
            complete_elliptic_integral = scp.ellipk(squared_modulus)
            elliptic_argument = (gamma - 2. * image_order * np.pi) / (2. * sqrt_term) - elliptic_integral_infinity + 2. * complete_elliptic_integral
        else:  # direct image
            elliptic_argument = gamma / (2. * sqrt_term) + elliptic_integral_infinity

        sine_jacobi, cn, dn, ph = scp.ellipj(elliptic_argument, squared_modulus)
        sine_squared = sine_jacobi * sine_jacobi

        term1 = -(q_parameter - periastron + 2. * black_hole_mass) / (4. * black_hole_mass * periastron)
        term2 = ((q_parameter - periastron + 6. * black_hole_mass) / (4. * black_hole_mass * periastron)) * sine_squared

        if verbose:
            print(f"Intermediate values: zeta_infinity: {zeta_infinity}, q_parameter: {q_parameter}, squared_modulus: {squared_modulus}, elliptic_integral_infinity: {elliptic_integral_infinity}, gamma: {gamma}")
            print(f"elliptic_argument: {elliptic_argument}, sine_jacobi: {sine_jacobi}, cn: {cn}, dn: {dn}, ph: {ph}")
            print(f"term1: {term1}, term2: {term2}")

        result = 1. - emission_radius * (term1 + term2)
        
        if verbose:
            print(f"calculate_luminet_equation_13 result: {result}")

        # Check for NaN result
        if np.isnan(result) and verbose:
            print(f"Result is NaN: {result}")

        return result

    @staticmethod
    def calculate_redshift_factor(radius: float, angle: float, inclination: float, black_hole_mass: float, impact_parameter: float) -> float:
        """Calculate the gravitational redshift factor (1 + z)."""
        min_radius = 3.01 * black_hole_mass  # Set argument minimum radius slightly larger than 3*black_hole_mass
        if radius < min_radius:
            radius = min_radius
            warnings.warn(f"Radius adjusted to minimum allowed value: {min_radius}")
        
        numerator = 1 + np.sqrt(black_hole_mass / radius**3) * impact_parameter * np.sin(inclination) * np.sin(angle)
        denominator = np.sqrt(1 - 3*black_hole_mass / radius)
        return numerator / denominator

    @staticmethod
    def calculate_intrinsic_flux(r_val: float, accretion_rate: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate the intrinsic flux of the accretion disk."""
        normalized_radius = r_val / black_hole_mass
        logarithm_argument = ((np.sqrt(normalized_radius) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / \
                  ((np.sqrt(normalized_radius) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
        result = (3. * black_hole_mass * accretion_rate / (8 * np.pi)) * (1 / ((normalized_radius - 3) * r_val ** 2.5)) * \
                 (np.sqrt(normalized_radius) - np.sqrt(6) + 3 ** -.5 * np.log10(logarithm_argument))
        if verbose:
            print(f"calculate_intrinsic_flux input: r_val: {r_val}, accretion_rate: {accretion_rate}, black_hole_mass: {black_hole_mass}")
            print(f"calculate_intrinsic_flux result: {result}")
        return result

    @staticmethod
    def calculate_observed_flux(r_val: float, accretion_rate: float, black_hole_mass: float, calculate_redshift_factor: float, verbose: bool = False) -> float:
        """Calculate the observed flux."""
        result = PhysicalFunctions.calculate_intrinsic_flux(r_val, accretion_rate, black_hole_mass, verbose) / calculate_redshift_factor**4
        if verbose:
            print(f"calculate_observed_flux input: r_val: {r_val}, accretion_rate: {accretion_rate}, black_hole_mass: {black_hole_mass}, calculate_redshift_factor: {calculate_redshift_factor}")
            print(f"calculate_observed_flux result: {result}")
        return result

    @staticmethod
    def calculate_phi_infinity(periastron: float, M: float, verbose: bool = False) -> float:
        """Calculate calculate_phi_infinity."""
        q_parameter = NumericalFunctions.calculate_q_parameter(periastron, M, verbose)
        ksq = NumericalFunctions.calculate_squared_elliptic_integral_modulus(periastron, M, verbose)
        zeta_infinity = NumericalFunctions.calculate_zeta_infinity(periastron, M, verbose)
        result = 2. * (np.sqrt(periastron / q_parameter)) * (scp.ellipk(ksq) - scp.ellipkinc(zeta_infinity, ksq))
        if verbose:
            print(f"calculate_phi_infinity input: periastron: {periastron}, M: {M}")
            print(f"calculate_phi_infinity result: {result}")
        return result

    @staticmethod
    def calculate_mu(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate calculate_mu."""
        result = float(2 * PhysicalFunctions.calculate_phi_infinity(periastron, black_hole_mass, verbose) - np.pi)
        if verbose:
            print(f"calculate_mu input: periastron: {periastron}, black_hole_mass: {black_hole_mass}")
            print(f"calculate_mu result: {result}")
        return result