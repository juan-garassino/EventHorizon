import numpy as np
from typing import Union, Optional, Callable, Dict, Any, Tuple, Iterable
import pandas as pd
import multiprocessing as mp
import logging
from .symbolic_expressions import SymbolicExpressions
from .numerical_functions import NumericalFunctions
from .physical_functions import PhysicalFunctions
from .geometric_functions import GeometricFunctions
from .impact_parameter import ImpactParameter
from .utilities import Utilities

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Simulation:
    @staticmethod
    def reorient_alpha(calculate_alpha: Union[float, np.ndarray], image_order: int, verbose: bool = False) -> np.ndarray:
        """Reorient the polar angle on the observation coordinate system."""
        logger.info(f"🔄 Reorienting alpha for image order {image_order}")
        calculate_alpha = np.asarray(calculate_alpha)  # Ensure calculate_alpha is a numpy array
        if verbose:
            logger.info(f"📊 Input alpha shape: {calculate_alpha.shape}")
        result = np.where(image_order > 0, (calculate_alpha + np.pi) % (2 * np.pi), calculate_alpha)
        if verbose:
            logger.info(f"📊 Output alpha shape: {result.shape}")
        return result

    @staticmethod
    def simulate_flux(
        calculate_alpha: np.ndarray,
        radius: float,
        theta_0: float,
        image_order: int,
        black_hole_mass: float,
        accretion_rate: float,
        objective_func: Optional[Callable] = None,
        root_kwargs: Optional[Dict[Any, Any]] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the bolometric flux for an accretion disk near a black hole."""
        logger.info(f"🌟 Simulating flux for radius {radius}, image order {image_order}")
        if objective_func is None:
            objective_func = Utilities.lambdify(("P", "calculate_alpha", "theta_0", "radius", "N", "M"), SymbolicExpressions.expr_r_inv())
        
        root_kwargs = root_kwargs if root_kwargs else {}
        
        # Ensure `calculate_alpha` is an array
        calculate_alpha = np.asarray(calculate_alpha)
        if verbose:
            logger.info(f"📊 Input alpha shape: {calculate_alpha.shape}")
        
        # Calculate impact parameter, redshift factor, and flux
        logger.info("📏 Calculating impact parameter")
        impact_parameters = np.asarray(ImpactParameter.calc_impact_parameter(calculate_alpha, radius, theta_0, image_order, black_hole_mass, **root_kwargs))
        if verbose:
            logger.info(f"📊 Impact parameters shape: {impact_parameters.shape}")
        
        logger.info("🌈 Calculating redshift factor")
        redshift_factors = np.asarray(PhysicalFunctions.calculate_redshift_factor(radius, calculate_alpha, theta_0, black_hole_mass, impact_parameters))
        if verbose:
            logger.info(f"📊 Redshift factors shape: {redshift_factors.shape}")
        
        logger.info("💫 Calculating observed flux")
        flux = np.asarray(PhysicalFunctions.calculate_observed_flux(radius, accretion_rate, black_hole_mass, redshift_factors))
        if verbose:
            logger.info(f"📊 Flux shape: {flux.shape}")
        
        # Reorient calculate_alpha and ensure all outputs are arrays
        logger.info("🔄 Reorienting alpha and finalizing output")
        reoriented_alpha = Simulation.reorient_alpha(calculate_alpha, image_order, verbose)
        if verbose:
            logger.info(f"📊 Final shapes - Alpha: {reoriented_alpha.shape}, Impact: {impact_parameters.shape}, Redshift: {redshift_factors.shape}, Flux: {flux.shape}")
        return reoriented_alpha, impact_parameters, redshift_factors, flux

    @staticmethod
    def _worker_function(calculate_alpha, radius, theta_0, image_order, black_hole_mass, accretion_rate, root_kwargs, verbose):
        logger.info(f"👷 Worker starting for radius {radius}, image order {image_order}")
        result = Simulation.simulate_flux(calculate_alpha, radius, theta_0, image_order, black_hole_mass, accretion_rate, None, root_kwargs, verbose)
        logger.info(f"✅ Worker completed for radius {radius}, image order {image_order}")
        return result

    @staticmethod
    def generate_image_data(
        calculate_alpha: np.ndarray,
        radii: Iterable[float],
        theta_0: float,
        image_orders: Iterable[int],
        black_hole_mass: float,
        accretion_rate: float,
        root_kwargs: Dict[str, Any],
        verbose: bool = False
    ) -> pd.DataFrame:
        """Generate the data needed to produce an image of a black hole."""
        logger.info("🖼️ Starting image data generation")
        if verbose:
            logger.info(f"📊 Input shapes - Alpha: {calculate_alpha.shape}, Radii: {len(radii)}, Image orders: {len(image_orders)}")
        
        data = []
        for image_order in image_orders:
            logger.info(f"🔄 Processing image order {image_order}")
            with mp.Pool(mp.cpu_count()) as pool:
                logger.info(f"🚀 Launching multiprocessing pool with {mp.cpu_count()} workers")
                arguments = [(calculate_alpha, radius, theta_0, image_order, black_hole_mass, accretion_rate, root_kwargs, verbose) for radius in radii]
                results = pool.starmap(Simulation._worker_function, arguments)
                
                for radius, (reoriented_angles, impact_parameters, redshift_factors, flux) in zip(radii, results):
                    logger.info(f"📊 Processing results for radius {radius}")
                    
                    if (isinstance(reoriented_angles, np.ndarray) and
                        isinstance(impact_parameters, np.ndarray) and
                        isinstance(redshift_factors, np.ndarray) and
                        isinstance(flux, np.ndarray)):
                        # Ensure all arrays are of the same length
                        min_len = min(len(reoriented_angles), len(impact_parameters), len(redshift_factors), len(flux))
                        
                        # Ensure consistency in length
                        reoriented_angles = reoriented_angles[:min_len]
                        impact_parameters = impact_parameters[:min_len]
                        redshift_factors = redshift_factors[:min_len]
                        flux = flux[:min_len]
                        
                        logger.info(f"📝 Adding {min_len} data points for radius {radius}")
                        if verbose:
                            logger.info(f"📊 Data shapes - Alpha: {reoriented_angles.shape}, Impact: {impact_parameters.shape}, Redshift: {redshift_factors.shape}, Flux: {flux.shape}")
                        
                        data.extend([
                            {
                                "calculate_alpha": alpha, "impact_parameters": b_val, "redshift_factors": opz_val,
                                "radius": radius, "image_order": image_order, "flux": flux_val,
                                "x_values": b_val * np.cos(alpha), "y_values": b_val * np.sin(alpha)
                            }
                            for alpha, b_val, opz_val, flux_val in zip(reoriented_angles, impact_parameters, redshift_factors, flux)
                        ])
                    else:
                        logger.error("❌ Error: One of the returned values is not an ndarray or has inconsistent dimensions")

        logger.info("✅ Image data generation complete")
        df = pd.DataFrame(data)
        if verbose:
            logger.info(f"📊 Final DataFrame shape: {df.shape}")
            logger.info(f"🔑 DataFrame columns: {df.columns}")
        return df