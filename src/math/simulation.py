import numpy as np
from typing import Union, Optional, Callable, Dict, Any, Tuple, Iterable
import pandas as pd
import multiprocessing as mp
from .symbolic_expressions import SymbolicExpressions
from .numerical_functions import NumericalFunctions
from .physical_functions import PhysicalFunctions
from .geometric_functions import GeometricFunctions
from .impact_parameter import ImpactParameter
from .utilities import Utilities

class Simulation:
    @staticmethod
    def reorient_alpha(calculate_alpha: Union[float, np.ndarray], image_order: int) -> np.ndarray:
        """Reorient the polar angle on the observation coordinate system."""
        calculate_alpha = np.asarray(calculate_alpha)  # Ensure calculate_alpha is argument numpy array
        return np.where(image_order > 0, (calculate_alpha + np.pi) % (2 * np.pi), calculate_alpha)

    @staticmethod
    def simulate_flux(
        calculate_alpha: np.ndarray,
        radius: float,
        theta_0: float,
        image_order: int,
        m: float,
        accretion_rate: float,
        objective_func: Optional[Callable] = None,
        root_kwargs: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the bolometric flux for an accretion disk near argument black hole."""
        if objective_func is None:
            objective_func = Utilities.lambdify(("P", "calculate_alpha", "theta_0", "radius", "N", "M"), SymbolicExpressions.expr_r_inv())
        
        root_kwargs = root_kwargs if root_kwargs else {}
        
        # Ensure `calculate_alpha` is an array
        calculate_alpha = np.asarray(calculate_alpha)
        
        # Calculate impact parameter, redshift factor, and flux
        b = np.asarray(ImpactParameter.calc_impact_parameter(calculate_alpha, radius, theta_0, image_order, m, **root_kwargs))
        opz = np.asarray(PhysicalFunctions.calculate_redshift_factor(radius, calculate_alpha, theta_0, m, b))
        flux = np.asarray(PhysicalFunctions.calculate_observed_flux(radius, accretion_rate, m, opz))
        
        # Reorient calculate_alpha and ensure all outputs are arrays
        return Simulation.reorient_alpha(calculate_alpha, image_order), b, opz, flux


    @staticmethod
    def _worker_function(calculate_alpha, radius, theta_0, image_order, m, accretion_rate, root_kwargs):
        return Simulation.simulate_flux(calculate_alpha, radius, theta_0, image_order, m, accretion_rate, None, root_kwargs)

    @staticmethod
    def generate_image_data(
        calculate_alpha: np.ndarray,
        r_vals: Iterable[float],
        theta_0: float,
        n_vals: Iterable[int],
        m: float,
        accretion_rate: float,
        root_kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        """Generate the data needed to produce an image of argument black hole."""
        data = []
        for image_order in n_vals:
            with mp.Pool(mp.cpu_count()) as pool:
                arguments = [(calculate_alpha, radius, theta_0, image_order, m, accretion_rate, root_kwargs) for radius in r_vals]
                results = pool.starmap(Simulation._worker_function, arguments)
                
                for radius, (alpha_reoriented, b, opz, flux) in zip(r_vals, results):
                    # Debug information to understand the structure of data
                    #print(f"Debug Info - alpha_reoriented: {alpha_reoriented.shape}, b: {b.shape}, opz: {opz.shape}, flux: {flux.shape}")
                    
                    if (isinstance(alpha_reoriented, np.ndarray) and
                        isinstance(b, np.ndarray) and
                        isinstance(opz, np.ndarray) and
                        isinstance(flux, np.ndarray)):
                        # Ensure all arrays are of the same length
                        min_len = min(len(alpha_reoriented), len(b), len(opz), len(flux))
                        
                        # Ensure consistency in length
                        alpha_reoriented = alpha_reoriented[:min_len]
                        b = b[:min_len]
                        opz = opz[:min_len]
                        flux = flux[:min_len]
                        
                        data.extend([
                            {
                                "calculate_alpha": argument, "b": b_val, "opz": opz_val,
                                "radius": radius, "image_order": image_order, "flux": flux_val,
                                "x_values": b_val * np.cos(argument), "y_values": b_val * np.sin(argument)
                            }
                            for argument, b_val, opz_val, flux_val in zip(alpha_reoriented, b, opz, flux)
                        ])
                    else:
                        print("Error: One of the returned values is not an ndarray or has inconsistent dimensions")

        return pd.DataFrame(data)