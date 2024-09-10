import numpy as np
import sympy as sp
import scipy.special as scp
import warnings
import logging
from .numerical_functions import NumericalFunctions
from .geometric_functions import GeometricFunctions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhysicalFunctions:
    @staticmethod
    def calculate_luminet_equation_13(periastron: float, emission_radius: float, emission_angle: float, black_hole_mass: float, inclination: float, image_order: int = 0, tolerance: float = 1e-6, verbose: bool = False) -> float:
        """Relation between radius (where photon was emitted in accretion disk), argument and P."""
        logger.info("🚀 Starting Luminet Equation 13 calculation")
        if verbose:
            logger.info(f"📊 Input parameters: periastron={periastron}, emission_radius={emission_radius}, emission_angle={emission_angle}, black_hole_mass={black_hole_mass}, inclination={inclination}, image_order={image_order}")

        zeta_infinity = NumericalFunctions.calculate_zeta_infinity(periastron, black_hole_mass, verbose=verbose)
        q_parameter = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose=verbose)
        squared_modulus = NumericalFunctions.calculate_squared_elliptic_integral_modulus(periastron, black_hole_mass, verbose=verbose)

        if q_parameter <= 0:
            logger.warning("⚠️ Invalid q_parameter value")
            return np.nan

        elliptic_integral_infinity = scp.ellipkinc(zeta_infinity, squared_modulus)
        gamma = np.arccos(GeometricFunctions.calculate_cos_gamma(emission_angle, inclination))

        sqrt_term = np.sqrt(periastron / q_parameter)
        if sqrt_term == 0:
            logger.warning("⚠️ Invalid sqrt_term value")
            return np.nan

        if image_order:
            complete_elliptic_integral = scp.ellipk(squared_modulus)
            elliptic_argument = (gamma - 2. * image_order * np.pi) / (2. * sqrt_term) - elliptic_integral_infinity + 2. * complete_elliptic_integral
        else:
            elliptic_argument = gamma / (2. * sqrt_term) + elliptic_integral_infinity

        sine_jacobi, cn, dn, ph = scp.ellipj(elliptic_argument, squared_modulus)
        sine_squared = sine_jacobi * sine_jacobi

        term1 = -(q_parameter - periastron + 2. * black_hole_mass) / (4. * black_hole_mass * periastron)
        term2 = ((q_parameter - periastron + 6. * black_hole_mass) / (4. * black_hole_mass * periastron)) * sine_squared

        if verbose:
            logger.info("🧮 Intermediate calculations completed")
            logger.info(f"📊 Shapes: zeta_infinity={np.shape(zeta_infinity)}, q_parameter={np.shape(q_parameter)}, squared_modulus={np.shape(squared_modulus)}")
            logger.info(f"📊 Shapes: elliptic_integral_infinity={np.shape(elliptic_integral_infinity)}, gamma={np.shape(gamma)}")
            logger.info(f"📊 Shapes: elliptic_argument={np.shape(elliptic_argument)}, sine_jacobi={np.shape(sine_jacobi)}")
            logger.info(f"📊 Shapes: term1={np.shape(term1)}, term2={np.shape(term2)}")

        result = 1. - emission_radius * (term1 + term2)
        
        logger.info("🎉 Luminet Equation 13 calculation completed")
        return result

    @staticmethod
    def calculate_redshift_factor(radius: float, angle: float, inclination: float, black_hole_mass: float, impact_parameter: float) -> float:
        """Calculate the gravitational redshift factor (1 + z)."""
        logger.info("🌈 Starting redshift factor calculation")
        min_radius = 3.01 * black_hole_mass
        if radius < min_radius:
            radius = min_radius
            logger.warning(f"⚠️ Radius adjusted to minimum allowed value: {min_radius}")
        
        numerator = 1 + np.sqrt(black_hole_mass / radius**3) * impact_parameter * np.sin(inclination) * np.sin(angle)
        denominator = np.sqrt(1 - 3*black_hole_mass / radius)
        result = numerator / denominator
        logger.info("🌈 Redshift factor calculation completed")
        return result

    @staticmethod
    def calculate_intrinsic_flux(r_val: float, accretion_rate: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate the intrinsic flux of the accretion disk."""
        logger.info("💫 Starting intrinsic flux calculation")
        if verbose:
            logger.info(f"📊 Input parameters: r_val={r_val}, accretion_rate={accretion_rate}, black_hole_mass={black_hole_mass}")

        normalized_radius = r_val / black_hole_mass
        logarithm_argument = ((np.sqrt(normalized_radius) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / \
                  ((np.sqrt(normalized_radius) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
        result = (3. * black_hole_mass * accretion_rate / (8 * np.pi)) * (1 / ((normalized_radius - 3) * r_val ** 2.5)) * \
                 (np.sqrt(normalized_radius) - np.sqrt(6) + 3 ** -.5 * np.log10(logarithm_argument))
        
        logger.info("💫 Intrinsic flux calculation completed")
        return result

    @staticmethod
    def calculate_observed_flux(r_val: float, accretion_rate: float, black_hole_mass: float, calculate_redshift_factor: float, verbose: bool = False) -> float:
        """Calculate the observed flux."""
        logger.info("🔭 Starting observed flux calculation")
        if verbose:
            logger.info(f"📊 Input parameters: r_val={r_val}, accretion_rate={accretion_rate}, black_hole_mass={black_hole_mass}, calculate_redshift_factor={calculate_redshift_factor}")

        result = PhysicalFunctions.calculate_intrinsic_flux(r_val, accretion_rate, black_hole_mass, verbose) / calculate_redshift_factor**4
        
        logger.info("🔭 Observed flux calculation completed")
        return result

    @staticmethod
    def calculate_phi_infinity(periastron: float, M: float, verbose: bool = False) -> float:
        """Calculate calculate_phi_infinity."""
        logger.info("🌌 Starting phi infinity calculation")
        if verbose:
            logger.info(f"📊 Input parameters: periastron={periastron}, M={M}")

        q_parameter = NumericalFunctions.calculate_q_parameter(periastron, M, verbose)
        ksq = NumericalFunctions.calculate_squared_elliptic_integral_modulus(periastron, M, verbose)
        zeta_infinity = NumericalFunctions.calculate_zeta_infinity(periastron, M, verbose)
        result = 2. * (np.sqrt(periastron / q_parameter)) * (scp.ellipk(ksq) - scp.ellipkinc(zeta_infinity, ksq))
        
        logger.info("🌌 Phi infinity calculation completed")
        return result

    @staticmethod
    def calculate_mu(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate calculate_mu."""
        logger.info("🔄 Starting mu calculation")
        if verbose:
            logger.info(f"📊 Input parameters: periastron={periastron}, black_hole_mass={black_hole_mass}")

        result = float(2 * PhysicalFunctions.calculate_phi_infinity(periastron, black_hole_mass, verbose) - np.pi)
        
        logger.info("🔄 Mu calculation completed")
        return result