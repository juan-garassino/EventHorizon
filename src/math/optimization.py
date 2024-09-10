import numpy as np
from typing import Callable, Dict, Any, Tuple, Optional
from .physical_functions import PhysicalFunctions

class Optimization:
    @staticmethod
    def apply_midpoint_method(function: Callable, arguments: Dict[str, Any], x_values: list, y_values: list, index: int, verbose: bool = False) -> Tuple[list, list, int]:
        """Implement the midpoint method for root finding."""
        new_x, new_y = x_values.copy(), y_values.copy()

        midpoint_x = [new_x[index], new_x[index + 1]]
        midpoint_x = np.mean(midpoint_x)
        new_x.insert(index + 1, midpoint_x)

        y_subset = [new_y[index], new_y[index + 1]]
        midpoint_y = function(periastron=midpoint_x, **arguments)
        new_y.insert(index + 1, midpoint_y)
        y_subset.insert(1, midpoint_y)
        sign_change_indices = np.where(np.diff(np.sign(y_subset)))[0]
        new_index = index + sign_change_indices[0] if len(sign_change_indices) > 0 else index

        if verbose:
            print(f"Midpoint method: x_values={new_x}, y_values={new_y}, new_index={new_index}")

        return new_x, new_y, new_index

    @staticmethod
    def refine_solution_with_midpoint_method(function: Callable, arguments: Dict[str, Any], x_values: list, y_values: list, index_of_sign_change: int, iteration_count: int, verbose: bool = False) -> float:
        """Improve solutions using the midpoint method."""
        new_x, new_y = x_values.copy(), y_values.copy()  # Make argument copy to avoid modifying the original lists
        new_index = index_of_sign_change
        for i in range(iteration_count):
            new_x, new_y, new_index = Optimization.apply_midpoint_method(function=function, arguments=arguments, x_values=new_x, y_values=new_y, index=new_index, verbose=verbose)
            if verbose:
                print(f"Iteration {i+1}: updated_periastron = {new_x[new_index]}")
        
        updated_periastron = new_x[new_index]
        if verbose:
            print(f"Final updated_periastron = {updated_periastron}")
        return updated_periastron

    @staticmethod
    def calculate_periastron(_r: float, inclination: float, azimuthal_angle: float, black_hole_mass: float, midpoint_iterations: int = 100, 
                        plot_inbetween: bool = False, image_order: int = 0, minimum_periastron: float = 1., initial_guess_count: int = 20, verbose: bool = False) -> Optional[float]:
        """Calculate the periastron for given parameters."""
        periastron_range = list(np.linspace(minimum_periastron, 2. * _r, initial_guess_count))
        y_subset = [PhysicalFunctions.calculate_luminet_equation_13(P_value, _r, azimuthal_angle, black_hole_mass, inclination, image_order) for P_value in periastron_range]
        index = np.where(np.diff(np.sign(y_subset)))[0]

        if verbose:
            print(f"periastron_range = {periastron_range}")
            print(f"y_subset = {y_subset}")
            print(f"index = {index}")

        if len(index) == 0:
            if verbose:
                print("No sign change, cannot find periastron")
            return None  # No sign change, cannot find periastron

        periastron_solution = periastron_range[index[0]] if len(index) > 0 else None

        if (periastron_solution is not None) and (not np.isnan(periastron_solution)):
            args_eq13 = {"emission_radius": _r, "emission_angle": azimuthal_angle, "black_hole_mass": black_hole_mass, "inclination": inclination, "image_order": image_order}
            periastron_solution = Optimization.refine_solution_with_midpoint_method(
                function=PhysicalFunctions.calculate_luminet_equation_13, arguments=args_eq13,
                x_values=periastron_range, y_values=y_subset, index_of_sign_change=index[0] if len(index) > 0 else 0,
                iteration_count=midpoint_iterations, verbose=verbose
            )
        return periastron_solution