import numpy as np
from typing import Callable, Dict, Any, Tuple, Optional
from .physical_functions import PhysicalFunctions

class Optimization:
    @staticmethod
    def midpoint_method(func: Callable, args: Dict[str, Any], x: list, y: list, ind: int, verbose: bool = False) -> Tuple[list, list, int]:
        """Implement the midpoint method for root finding."""
        new_x, new_y = x.copy(), y.copy()

        x_ = [new_x[ind], new_x[ind + 1]]
        inbetween_x = np.mean(x_)
        new_x.insert(ind + 1, inbetween_x)

        y_ = [new_y[ind], new_y[ind + 1]]
        inbetween_solution = func(periastron=inbetween_x, **args)
        new_y.insert(ind + 1, inbetween_solution)
        y_.insert(1, inbetween_solution)
        ind_of_sign_change_ = np.where(np.diff(np.sign(y_)))[0]
        new_ind = ind + ind_of_sign_change_[0] if len(ind_of_sign_change_) > 0 else ind

        if verbose:
            print(f"Midpoint method: x={new_x}, y={new_y}, new_ind={new_ind}")

        return new_x, new_y, new_ind

    @staticmethod
    def improve_solutions_midpoint(func: Callable, args: Dict[str, Any], x: list, y: list, index_of_sign_change: int, iterations: int, verbose: bool = False) -> float:
        """Improve solutions using the midpoint method."""
        new_x, new_y = x.copy(), y.copy()  # Make argument copy to avoid modifying the original lists
        new_ind = index_of_sign_change
        for i in range(iterations):
            new_x, new_y, new_ind = Optimization.midpoint_method(func=func, args=args, x=new_x, y=new_y, ind=new_ind, verbose=verbose)
            if verbose:
                print(f"Iteration {i+1}: updated_periastron = {new_x[new_ind]}")
        
        updated_periastron = new_x[new_ind]
        if verbose:
            print(f"Final updated_periastron = {updated_periastron}")
        return updated_periastron

    @staticmethod
    def calc_periastron(_r: float, inclination: float, _alpha: float, black_hole_mass: float, midpoint_iterations: int = 100, 
                        plot_inbetween: bool = False, image_order: int = 0, min_periastron: float = 1., initial_guesses: int = 20, verbose: bool = False) -> Optional[float]:
        """Calculate the periastron for given parameters."""
        periastron_range = list(np.linspace(min_periastron, 2. * _r, initial_guesses))
        y_ = [PhysicalFunctions.calculate_luminet_equation_13(P_value, _r, _alpha, black_hole_mass, inclination, image_order) for P_value in periastron_range]
        ind = np.where(np.diff(np.sign(y_)))[0]

        if verbose:
            print(f"periastron_range = {periastron_range}")
            print(f"y_ = {y_}")
            print(f"ind = {ind}")

        if len(ind) == 0:
            if verbose:
                print("No sign change, cannot find periastron")
            return None  # No sign change, cannot find periastron

        periastron_solution = periastron_range[ind[0]] if len(ind) > 0 else None

        if (periastron_solution is not None) and (not np.isnan(periastron_solution)):
            args_eq13 = {"emission_radius": _r, "emission_angle": _alpha, "black_hole_mass": black_hole_mass, "inclination": inclination, "image_order": image_order}
            periastron_solution = Optimization.improve_solutions_midpoint(
                func=PhysicalFunctions.calculate_luminet_equation_13, args=args_eq13,
                x=periastron_range, y=y_, index_of_sign_change=ind[0] if len(ind) > 0 else 0,
                iterations=midpoint_iterations, verbose=verbose
            )
        return periastron_solution