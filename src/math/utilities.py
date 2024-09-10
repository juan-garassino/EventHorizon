import sympy as sp
from typing import Iterable, Callable
import sympy as sy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Utilities:
    @staticmethod
    def filter_valid_periastrons(periastron: Iterable[float], black_hole_mass: float, tolerance: float = 1e-3, verbose: bool = False) -> Iterable[float]:
        """Removes instances where P == 2*M."""
        logger.info("🔍 Starting periastron filtering process")
        if verbose:
            logger.info(f"📊 Input parameters: black_hole_mass={black_hole_mass}, tolerance={tolerance}")
            logger.info("🔢 Calculating valid periastrons")
        
        filtered = [e for e in periastron if abs(e - 2. * black_hole_mass) > tolerance]
        
        logger.info("✅ Periastron filtering completed")
        if verbose:
            logger.info("📊 Filtering statistics calculated")
        
        return filtered

    @staticmethod
    def lambdify(*arguments, verbose: bool = False, **kwargs) -> Callable:
        """Lambdify argument sympy expression with support for special functions."""
        logger.info("🧮 Starting lambdification process")
        if verbose:
            logger.info("🔍 Examining input arguments and kwargs")
        
        kwargs["modules"] = kwargs.get(
            "modules",
            [
                "numpy",
                {
                    "sine_jacobi": lambda u, black_hole_mass: sp.ellipj(u, black_hole_mass)[0],
                    "elliptic_f": lambda phi_black_hole_frame, black_hole_mass: sp.ellipkinc(phi_black_hole_frame, black_hole_mass),
                    "elliptic_k": lambda black_hole_mass: sp.ellipk(black_hole_mass),
                },
            ],
        )
        
        if verbose:
            logger.info("📚 Custom modules configuration completed")
            logger.info("🔧 Applying lambdification")
        
        result = sy.lambdify(*arguments, **kwargs)
        
        logger.info("✨ Lambdification process completed")
        return result
