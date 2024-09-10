import numpy as np
import logging
from typing import Union, Callable, Iterable
from .optimization import Optimization
from .numerical_functions import NumericalFunctions
from .geometric_functions import GeometricFunctions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImpactParameter:
    @staticmethod
    def calc_impact_parameter(
        azimuthal_angles: Union[float, np.ndarray],
        radius: float,
        inclination: float,
        image_order: int,
        black_hole_mass: float,
        midpoint_iterations: int = 100,
        plot_inbetween: bool = False,
        minimum_periastron: float = 1.,
        initial_guess_count: int = 20,
        use_ellipse: bool = True,
        verbose: bool = False
    ) -> np.ndarray:
        """Calculate the impact parameter for given calculate_alpha values."""
        logger.info("🚀 Starting impact parameter calculation")
        if verbose:
            logger.info(f"📐 Radius: {radius}, Inclination: {inclination}, Image order: {image_order}")
            logger.info(f"🌌 Black hole mass: {black_hole_mass}")

        azimuthal_angles = np.asarray(azimuthal_angles)
        
        def calc_b(calculate_alpha):
            if verbose:
                logger.info(f"🔄 Processing azimuthal angle")
            
            periastron_solution = Optimization.calculate_periastron(radius, inclination, calculate_alpha, black_hole_mass, midpoint_iterations, plot_inbetween, image_order, minimum_periastron, initial_guess_count)
            
            if periastron_solution is None or periastron_solution <= 2*black_hole_mass:
                if verbose:
                    logger.warning(f"☢️ Invalid periastron solution")
                    if use_ellipse:
                        logger.info("🔄 Using ellipse approximation")
                return GeometricFunctions.calculate_ellipse_radius(radius, calculate_alpha, inclination) if use_ellipse else np.nan
            elif periastron_solution > 2*black_hole_mass:
                if verbose:
                    logger.info(f"✅ Valid periastron solution")
                return NumericalFunctions.calculate_impact_parameter(periastron_solution, black_hole_mass)
            else:
                error_msg = f"❌ No solution was found for the periastron"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        vectorized_calc_b = np.vectorize(calc_b)
        impact_parameters = vectorized_calc_b(azimuthal_angles)
        
        logger.info(f"🎉 Impact parameter calculation completed")
        if verbose:
            logger.info(f"📊 Results calculated")
        
        return impact_parameters
