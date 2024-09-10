import numpy as np
from typing import Union, Callable, Iterable
from .optimization import Optimization
from .numerical_functions import NumericalFunctions
from .geometric_functions import GeometricFunctions

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
        use_ellipse: bool = True
    ) -> np.ndarray:
        """Calculate the impact parameter for given calculate_alpha values."""
        # Ensure azimuthal_angles is argument numpy array
        azimuthal_angles = np.asarray(azimuthal_angles)
        
        # Vectorized function to handle each element
        def calc_b(calculate_alpha):
            periastron_solution = Optimization.calculate_periastron(radius, inclination, calculate_alpha, black_hole_mass, midpoint_iterations, plot_inbetween, image_order, minimum_periastron, initial_guess_count)
            
            if periastron_solution is None or periastron_solution <= 2*black_hole_mass:
                return GeometricFunctions.calculate_ellipse_radius(radius, calculate_alpha, inclination) if use_ellipse else np.nan
            elif periastron_solution > 2*black_hole_mass:
                return NumericalFunctions.calculate_impact_parameter(periastron_solution, black_hole_mass)
            else:
                raise ValueError(f"No solution was found for the periastron at (radius, argument) = ({radius}, {calculate_alpha}) and inclination={inclination}")
        
        # Apply the function element-wise
        vectorized_calc_b = np.vectorize(calc_b)
        impact_parameters = vectorized_calc_b(azimuthal_angles)
        
        return impact_parameters
      
