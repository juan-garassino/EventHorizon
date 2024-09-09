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
    def reorient_alpha(alpha: Union[float, np.ndarray], n: int) -> np.ndarray:
        """Reorient the polar angle on the observation coordinate system."""
        alpha = np.asarray(alpha)  # Ensure alpha is a numpy array
        return np.where(n > 0, (alpha + np.pi) % (2 * np.pi), alpha)

    @staticmethod
    def simulate_flux(
        alpha: np.ndarray,
        r: float,
        theta_0: float,
        n: int,
        m: float,
        acc: float,
        objective_func: Optional[Callable] = None,
        root_kwargs: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the bolometric flux for an accretion disk near a black hole."""
        if objective_func is None:
            objective_func = Utilities.lambdify(("P", "alpha", "theta_0", "r", "N", "M"), SymbolicExpressions.expr_r_inv())
        
        root_kwargs = root_kwargs if root_kwargs else {}
        
        # Ensure `alpha` is an array
        alpha = np.asarray(alpha)
        
        # Calculate impact parameter, redshift factor, and flux
        b = np.asarray(ImpactParameter.calc_impact_parameter(alpha, r, theta_0, n, m, **root_kwargs))
        opz = np.asarray(PhysicalFunctions.redshift_factor(r, alpha, theta_0, m, b))
        flux = np.asarray(PhysicalFunctions.flux_observed(r, acc, m, opz))
        
        # Reorient alpha and ensure all outputs are arrays
        return Simulation.reorient_alpha(alpha, n), b, opz, flux


    @staticmethod
    def _worker_function(alpha, r, theta_0, n, m, acc, root_kwargs):
        return Simulation.simulate_flux(alpha, r, theta_0, n, m, acc, None, root_kwargs)

    @staticmethod
    def generate_image_data(
        alpha: np.ndarray,
        r_vals: Iterable[float],
        theta_0: float,
        n_vals: Iterable[int],
        m: float,
        acc: float,
        root_kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        """Generate the data needed to produce an image of a black hole."""
        data = []
        for n in n_vals:
            with mp.Pool(mp.cpu_count()) as pool:
                args = [(alpha, r, theta_0, n, m, acc, root_kwargs) for r in r_vals]
                results = pool.starmap(Simulation._worker_function, args)
                
                for r, (alpha_reoriented, b, opz, flux) in zip(r_vals, results):
                    # Debug information to understand the structure of data
                    print(f"Debug Info - alpha_reoriented: {alpha_reoriented.shape}, b: {b.shape}, opz: {opz.shape}, flux: {flux.shape}")
                    
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
                                "alpha": a, "b": b_val, "opz": opz_val,
                                "r": r, "n": n, "flux": flux_val,
                                "x": b_val * np.cos(a), "y": b_val * np.sin(a)
                            }
                            for a, b_val, opz_val, flux_val in zip(alpha_reoriented, b, opz, flux)
                        ])
                    else:
                        print("Error: One of the returned values is not an ndarray or has inconsistent dimensions")

        return pd.DataFrame(data)