import numpy as np
from typing import Union, Callable, Iterable
from .optimization import Optimization
from .numerical_functions import NumericalFunctions
from .geometric_functions import GeometricFunctions

class ImpactParameter:
    @staticmethod
    def calc_impact_parameter(
        alpha_val: Union[float, np.ndarray],
        r_value: float,
        theta_0_val: float,
        n: int,
        m: float,
        midpoint_iterations: int = 100,
        plot_inbetween: bool = False,
        min_periastron: float = 1.,
        initial_guesses: int = 20,
        use_ellipse: bool = True
    ) -> np.ndarray:
        """Calculate the impact parameter for given alpha values."""
        # Ensure alpha_val is a numpy array
        alpha_val = np.asarray(alpha_val)
        
        # Vectorized function to handle each element
        def calc_b(alpha):
            periastron_solution = Optimization.calc_periastron(r_value, theta_0_val, alpha, m, midpoint_iterations, plot_inbetween, n, min_periastron, initial_guesses)
            
            if periastron_solution is None or periastron_solution <= 2*m:
                return GeometricFunctions.ellipse(r_value, alpha, theta_0_val) if use_ellipse else np.nan
            elif periastron_solution > 2*m:
                return NumericalFunctions.calc_b_from_periastron(periastron_solution, m)
            else:
                raise ValueError(f"No solution was found for the periastron at (r, a) = ({r_value}, {alpha}) and incl={theta_0_val}")
        
        # Apply the function element-wise
        vectorized_calc_b = np.vectorize(calc_b)
        b = vectorized_calc_b(alpha_val)
        
        return b
      
class Utilities:
    @staticmethod
    def filter_periastrons(periastron: Iterable[float], bh_mass: float, tol: float = 1e-3) -> Iterable[float]:
        """Removes instances where P == 2*M."""
        return [e for e in periastron if abs(e - 2. * bh_mass) > tol]

    @staticmethod
    def lambdify(*args, **kwargs) -> Callable:
        """Lambdify a sympy expression with support for special functions."""
        kwargs["modules"] = kwargs.get(
            "modules",
            [
                "numpy",
                {
                    "sn": lambda u, m: sp.ellipj(u, m)[0],
                    "elliptic_f": lambda phi, m: sp.ellipkinc(phi, m),
                    "elliptic_k": lambda m: sp.ellipk(m),
                },
            ],
        )
        return sy.lambdify(*args, **kwargs)
