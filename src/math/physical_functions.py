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

        z_inf = NumericalFunctions.calculate_zeta_infinity(periastron, black_hole_mass, verbose=verbose)
        q = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose=verbose)
        m_ = NumericalFunctions.calculate_squared_elliptic_integral_modulus(periastron, black_hole_mass, verbose=verbose)

        # Check for invalid q
        if q <= 0:
            if verbose:
                print(f"Invalid q value: {q}")
            return np.nan

        # Use elliptic_f instead of ellipkinc
        ell_inf = scp.ellipkinc(z_inf, m_)
        g = np.arccos(GeometricFunctions.cos_gamma(emission_angle, inclination))

        # Check for division by zero or invalid values
        sqrt_term = np.sqrt(periastron / q)
        if sqrt_term == 0:
            if verbose:
                print(f"Invalid sqrt_term value: {sqrt_term}")
            return np.nan

        if image_order:  # higher order image
            ell_k = scp.ellipk(m_)
            ellips_arg = (g - 2. * image_order * np.pi) / (2. * sqrt_term) - ell_inf + 2. * ell_k
        else:  # direct image
            ellips_arg = g / (2. * sqrt_term) + ell_inf

        sn, cn, dn, ph = scp.ellipj(ellips_arg, m_)
        sn2 = sn * sn

        term1 = -(q - periastron + 2. * black_hole_mass) / (4. * black_hole_mass * periastron)
        term2 = ((q - periastron + 6. * black_hole_mass) / (4. * black_hole_mass * periastron)) * sn2

        if verbose:
            print(f"Intermediate values: z_inf: {z_inf}, q: {q}, m_: {m_}, ell_inf: {ell_inf}, g: {g}")
            print(f"ellips_arg: {ellips_arg}, sn: {sn}, cn: {cn}, dn: {dn}, ph: {ph}")
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
        r_ = r_val / black_hole_mass
        log_arg = ((np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / \
                  ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
        result = (3. * black_hole_mass * accretion_rate / (8 * np.pi)) * (1 / ((r_ - 3) * r_val ** 2.5)) * \
                 (np.sqrt(r_) - np.sqrt(6) + 3 ** -.5 * np.log10(log_arg))
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
        q = NumericalFunctions.calculate_q_parameter(periastron, M, verbose)
        ksq = NumericalFunctions.calculate_squared_elliptic_integral_modulus(periastron, M, verbose)
        z_inf = NumericalFunctions.calculate_zeta_infinity(periastron, M, verbose)
        result = 2. * (np.sqrt(periastron / q)) * (scp.ellipk(ksq) - scp.ellipkinc(z_inf, ksq))
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