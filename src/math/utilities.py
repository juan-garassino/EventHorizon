import sympy as sp
from typing import Iterable, Callable
import sympy as sy

class Utilities:
    @staticmethod
    def filter_valid_periastrons(periastron: Iterable[float], black_hole_mass: float, tolerance: float = 1e-3) -> Iterable[float]:
        """Removes instances where P == 2*M."""
        return [e for e in periastron if abs(e - 2. * black_hole_mass) > tolerance]

    @staticmethod
    def lambdify(*arguments, **kwargs) -> Callable:
        """Lambdify argument sympy expression with support for special functions."""
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
        return sy.lambdify(*arguments, **kwargs)
