import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumericalFunctions:
    @staticmethod
    def validate_parameters(periastron: float, black_hole_mass: float, verbose: bool = False) -> bool:
        """Validate the input parameters."""
        logger.info("🔍 Validating parameters")
        if periastron <= 2 * black_hole_mass:
            logger.warning(f"☢️ Invalid periastron for black_hole_mass")
            return False
        if black_hole_mass <= 0:
            logger.warning(f"☢️ Invalid black_hole_mass")
            return False
        if verbose:
            logger.info(f"✅ Parameters valid")
        return True

    @staticmethod
    def calculate_q_parameter(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Convert Periastron distance P to the variable Q."""
        logger.info("🧮 Calculating Q parameter")
        if not NumericalFunctions.validate_parameters(periastron, black_hole_mass, verbose):
            return np.nan
        if verbose:
            logger.info(f"🔢 Q parameter calculation in progress")
        return np.sqrt((periastron - 2 * black_hole_mass) * (periastron + 6 * black_hole_mass))

    @staticmethod
    def calculate_impact_parameter(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Get impact parameter from Periastron distance P."""
        logger.info("💥 Calculating impact parameter")
        if not NumericalFunctions.validate_parameters(periastron, black_hole_mass, verbose):
            return np.nan
        if verbose:
            logger.info(f"🔢 Impact parameter calculation in progress")
        return np.sqrt((periastron**3) / (periastron - 2 * black_hole_mass))

    @staticmethod
    def calculate_elliptic_integral_modulus(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate modulus of elliptic integral."""
        logger.info("🔄 Calculating elliptic integral modulus")
        q_parameter = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose)
        if np.isnan(q_parameter) or q_parameter == 0:
            logger.warning("☢️ Invalid q_parameter value or division by zero")
            return np.nan
        if verbose:
            logger.info(f"🔢 Elliptic integral modulus calculation in progress")
        return np.sqrt((q_parameter - periastron + 6 * black_hole_mass) / (2 * q_parameter))

    @staticmethod
    def calculate_squared_elliptic_integral_modulus(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate the squared modulus of elliptic integral."""
        logger.info("🔢 Calculating squared elliptic integral modulus")
        q_parameter = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose)
        if np.isnan(q_parameter) or q_parameter == 0:
            logger.warning("☢️ Invalid q_parameter value or division by zero")
            return np.nan
        if verbose:
            logger.info(f"🔢 Squared elliptic integral modulus calculation in progress")
        return (q_parameter - periastron + 6 * black_hole_mass) / (2 * q_parameter)

    @staticmethod
    def calculate_zeta_infinity(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate Zeta_inf for elliptic integral F(Zeta_inf, calculate_elliptic_integral_modulus)."""
        logger.info("🔄 Calculating Zeta infinity")
        q_parameter = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose)
        if np.isnan(q_parameter) or (q_parameter - periastron + 6 * black_hole_mass) == 0:
            logger.warning("☢️ Invalid q_parameter value or division by zero")
            return np.nan
        arg = (q_parameter - periastron + 2 * black_hole_mass) / (q_parameter - periastron + 6 * black_hole_mass)
        if not (0 <= arg <= 1):
            logger.warning(f"☢️ Invalid argument for asin")
            return np.nan
        if verbose:
            logger.info(f"🔢 Zeta infinity calculation in progress")
        return np.arcsin(np.sqrt(arg))

    @staticmethod
    def calculate_zeta_radius(periastron: float, radius: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate the elliptic integral argument Zeta_r for argument given value of P and radius."""
        logger.info("🔄 Calculating Zeta radius")
        q_parameter = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose)
        if not NumericalFunctions.validate_parameters(periastron, black_hole_mass, verbose) or np.isnan(q_parameter):
            return np.nan
        denominator = q_parameter - periastron + 6 * black_hole_mass
        if denominator == 0:
            logger.warning("☢️ Division by zero")
            return np.nan
        argument = (q_parameter - periastron + 2 * black_hole_mass + (4 * black_hole_mass * periastron) / radius) / denominator
        if not (0 <= argument <= 1):
            logger.warning(f"☢️ Invalid argument for asin")
            return np.nan
        if verbose:
            logger.info(f"🔢 Zeta radius calculation in progress")
        return np.arcsin(np.sqrt(argument))