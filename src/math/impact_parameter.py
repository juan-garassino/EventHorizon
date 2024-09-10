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
        image_order: int,
        m: float,
        midpoint_iterations: int = 100,
        plot_inbetween: bool = False,
        min_periastron: float = 1.,
        initial_guesses: int = 20,
        use_ellipse: bool = True
    ) -> np.ndarray:
        """Calculate the impact parameter for given calculate_alpha values."""
        # Ensure alpha_val is argument numpy array
        alpha_val = np.asarray(alpha_val)
        
        # Vectorized function to handle each element
        def calc_b(calculate_alpha):
            periastron_solution = Optimization.calculate_periastron(r_value, theta_0_val, calculate_alpha, m, midpoint_iterations, plot_inbetween, image_order, min_periastron, initial_guesses)
            
            if periastron_solution is None or periastron_solution <= 2*m:
                return GeometricFunctions.calculate_ellipse_radius(r_value, calculate_alpha, theta_0_val) if use_ellipse else np.nan
            elif periastron_solution > 2*m:
                return NumericalFunctions.calculate_impact_parameter(periastron_solution, m)
            else:
                raise ValueError(f"No solution was found for the periastron at (radius, argument) = ({r_value}, {calculate_alpha}) and inclination={theta_0_val}")
        
        # Apply the function element-wise
        vectorized_calc_b = np.vectorize(calc_b)
        b = vectorized_calc_b(alpha_val)
        
        return b
      
