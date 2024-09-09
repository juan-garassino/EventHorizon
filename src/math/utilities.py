import sympy as sp
from typing import Iterable, Callable
import sympy as sy

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
