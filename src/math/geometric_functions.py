import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeometricFunctions:
    @staticmethod
    def calculate_cos_gamma(azimuthal_angle: float, inclination: float, tolerance: float = 1e-5, verbose: bool = False) -> float:
        """Calculate the cos of the angle gamma."""
        logger.info("📐 Calculating cos(gamma)")
        if verbose:
            logger.info(f"🔍 Input parameters: azimuthal_angle={azimuthal_angle}, inclination={inclination}")
        
        if abs(inclination) < tolerance:
            logger.info("⚠️ Inclination close to zero")
            return 0
        
        result = np.cos(azimuthal_angle) / np.sqrt(np.cos(azimuthal_angle) ** 2 + 1 / (np.tan(inclination) ** 2))
        
        logger.info("✅ cos(gamma) calculation complete")
        return result

    @staticmethod
    def calculate_cos_alpha(phi_black_hole_frame: float, inclination: float, verbose: bool = False) -> float:
        """Returns cos(angle) calculate_alpha in observer frame given angles phi_black_hole_frame (black hole frame) and inclination (black hole frame)."""
        logger.info("📐 Calculating cos(alpha)")
        if verbose:
            logger.info(f"🔍 Input parameters: phi_black_hole_frame={phi_black_hole_frame}, inclination={inclination}")
        
        result = np.cos(phi_black_hole_frame) * np.cos(inclination) / np.sqrt((1 - np.sin(inclination) ** 2 * np.cos(phi_black_hole_frame) ** 2))
        
        logger.info("✅ cos(alpha) calculation complete")
        return result

    @staticmethod
    def calculate_alpha(phi_black_hole_frame: float, inclination: float, verbose: bool = False) -> float:
        """Returns observer coordinate of photon given phi_black_hole_frame (BHF) and inclination (BHF)."""
        logger.info("📐 Calculating alpha")
        if verbose:
            logger.info(f"🔍 Input parameters: phi_black_hole_frame={phi_black_hole_frame}, inclination={inclination}")
        
        cos_alpha = GeometricFunctions.calculate_cos_alpha(phi_black_hole_frame, inclination, verbose)
        result = np.arccos(cos_alpha)
        
        logger.info("✅ alpha calculation complete")
        return result

    @staticmethod
    def calculate_ellipse_radius(radius: float, argument: float, inclination: float, verbose: bool = False) -> float:
        """Equation of an calculate_ellipse_radius, reusing the definition of calculate_cos_gamma."""
        logger.info("📏 Calculating ellipse radius")
        if verbose:
            logger.info(f"🔍 Input parameters: radius={radius}, argument={argument}, inclination={inclination}")
        
        gamma = np.arccos(GeometricFunctions.calculate_cos_gamma(argument, inclination, verbose=verbose))
        result = radius * np.sin(gamma)
        
        logger.info("✅ Ellipse radius calculation complete")
        return result