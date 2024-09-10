import numpy as np
from typing import Callable, Dict, Any, Tuple, Optional
from .physical_functions import PhysicalFunctions

class Optimization:
    @staticmethod
    def apply_midpoint_method(function: Callable, arguments: Dict[str, Any], x_values: list, y_values: list, index: int, verbose: bool = False) -> Tuple[list, list, int]:
        """Implement the midpoint method for root finding."""
        new_x, new_y = x_values.copy(), y_values.copy()

        x_ = [new_x[index], new_x[index + 1]]
        inbetween_x = np.mean(x_)
        new_x.insert(index + 1, inbetween_x)

        y_ = [new_y[index], new_y[index + 1]]
        inbetween_solution = function(periastron=inbetween_x, **arguments)
        new_y.insert(index + 1, inbetween_solution)
        y_.insert(1, inbetween_solution)
        ind_of_sign_change_ = np.where(np.diff(np.sign(y_)))[0]
        new_ind = index + ind_of_sign_change_[0] if len(ind_of_sign_change_) > 0 else index

        if verbose:
            print(f"Midpoint method: x_values={new_x}, y_values={new_y}, new_ind={new_ind}")

        return new_x, new_y, new_ind

    @staticmethod
    def refine_solution_with_midpoint_method(function: Callable, arguments: Dict[str, Any], x_values: list, y_values: list, index_of_sign_change: int, iteration_count: int, verbose: bool = False) -> float:
        """Improve solutions using the midpoint method."""
        new_x, new_y = x_values.copy(), y_values.copy()  # Make argument copy to avoid modifying the original lists
        new_ind = index_of_sign_change
        for i in range(iteration_count):
            new_x, new_y, new_ind = Optimization.apply_midpoint_method(function=function, arguments=arguments, x_values=new_x, y_values=new_y, index=new_ind, verbose=verbose)
            if verbose:
                print(f"Iteration {i+1}: updated_periastron = {new_x[new_ind]}")
        
        updated_periastron = new_x[new_ind]
        if verbose:
            print(f"Final updated_periastron = {updated_periastron}")
        return updated_periastron

    @staticmethod
    def calculate_periastron(_r: float, inclination: float, _alpha: float, black_hole_mass: float, midpoint_iterations: int = 100, 
                        plot_inbetween: bool = False, image_order: int = 0, min_periastron: float = 1., initial_guesses: int = 20, verbose: bool = False) -> Optional[float]:
        """Calculate the periastron for given parameters."""
        periastron_range = list(np.linspace(min_periastron, 2. * _r, initial_guesses))
        y_ = [PhysicalFunctions.calculate_luminet_equation_13(P_value, _r, _alpha, black_hole_mass, inclination, image_order) for P_value in periastron_range]
        index = np.where(np.diff(np.sign(y_)))[0]

        if verbose:
            print(f"periastron_range = {periastron_range}")
            print(f"y_ = {y_}")
            print(f"index = {index}")

        if len(index) == 0:
            if verbose:
                print("No sign change, cannot find periastron")
            return None  # No sign change, cannot find periastron

        periastron_solution = periastron_range[index[0]] if len(index) > 0 else None

        if (periastron_solution is not None) and (not np.isnan(periastron_solution)):
            args_eq13 = {"emission_radius": _r, "emission_angle": _alpha, "black_hole_mass": black_hole_mass, "inclination": inclination, "image_order": image_order}
            periastron_solution = Optimization.refine_solution_with_midpoint_method(
                function=PhysicalFunctions.calculate_luminet_equation_13, arguments=args_eq13,
                x_values=periastron_range, y_values=y_, index_of_sign_change=index[0] if len(index) > 0 else 0,
                iteration_count=midpoint_iterations, verbose=verbose
            )
        return periastron_solution