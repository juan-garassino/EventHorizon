import sympy as sy
import scipy.special as sp
import scipy.optimize as sopt
import numpy.typing as npt
import pandas as pd
import multiprocessing as mp
from typing import Callable, Union, Tuple, Optional, Dict, Any, Iterable
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors

# Constants and symbols
M, P, Q, r, theta_0, alpha, b, gamma, N, z_op, F_s, mdot = sy.symbols("M P Q r theta_0 alpha b gamma N z_op F_s mdot")

class SymbolicExpressions:
    @staticmethod
    @lru_cache(maxsize=None)
    def expr_q() -> sy.Expr:
        """Generate a sympy expression for Q."""
        return sy.sqrt((P - 2*M) * (P + 6*M))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_b() -> sy.Expr:
        """Generate a sympy expression for b, the radial coordinate in the observer's frame."""
        return sy.sqrt((P**3) / (P - 2*M))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_k() -> sy.Expr:
        """Generate a sympy expression for k; k**2 is used as a modulus in the elliptic integrals."""
        return sy.sqrt((Q - P + 6*M) / (2*Q))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_zeta_inf() -> sy.Expr:
        """Generate a sympy expression for zeta_inf."""
        return sy.asin(sy.sqrt((Q - P + 2*M) / (Q - P + 6*M)))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_gamma() -> sy.Expr:
        """Generate a sympy expression for gamma, an angle that relates alpha and theta_0."""
        return sy.acos(sy.cos(alpha) / sy.sqrt(sy.cos(alpha)**2 + sy.tan(theta_0)**-2))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_u() -> sy.Expr:
        """Generate a sympy expression for the argument to sn in equation 13 of Luminet (1977)."""
        zeta_inf, k = SymbolicExpressions.expr_zeta_inf(), SymbolicExpressions.expr_k()
        return sy.Piecewise(
            (gamma / (2 * sy.sqrt(P / Q)) + sy.elliptic_f(zeta_inf, k**2), sy.Eq(N, 0)),
            (
                (gamma - 2 * N * sy.pi) / (2 * sy.sqrt(P / Q))
                - sy.elliptic_f(zeta_inf, k**2)
                + 2 * sy.elliptic_k(k**2),
                True,
            ),
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_r_inv() -> sy.Expr:
        """Generate a sympy expression for 1/r."""
        Q, k = SymbolicExpressions.expr_q(), SymbolicExpressions.expr_k()
        sn = sy.Function("sn")
        u = SymbolicExpressions.expr_u()
        return (1 / (4*M*P)) * (-(Q - P + 2*M) + (Q - P + 6*M) * sn(u, k**2)**2)

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_one_plus_z() -> sy.Expr:
        """Generate an expression for the redshift 1+z."""
        return (1 + sy.sqrt(M / r**3) * b * sy.sin(theta_0) * sy.sin(alpha)) / sy.sqrt(1 - 3*M / r)

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_fs() -> sy.Expr:
        """Generate an expression for the flux of an accreting disk."""
        rstar = r / M
        return (
            ((3 * M * mdot) / (8 * sy.pi))
            * (1 / ((rstar - 3) * rstar ** (5 / 2)))
            * (
                sy.sqrt(rstar)
                - sy.sqrt(6)
                + (sy.sqrt(3) / 3)
                * sy.ln(
                    ((sy.sqrt(rstar) + sy.sqrt(3)) * (sy.sqrt(6) - sy.sqrt(3)))
                    / ((sy.sqrt(rstar) - sy.sqrt(3)) * (sy.sqrt(6) + sy.sqrt(3)))
                )
            )
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_f0() -> sy.Expr:
        """Generate an expression for the observed bolometric flux."""
        return F_s / z_op**4

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_f0_normalized() -> sy.Expr:
        """Generate an expression for the normalized observed bolometric flux."""
        return SymbolicExpressions.expr_f0() / ((8 * sy.pi) / (3 * M * mdot))

    @staticmethod
    @lru_cache(maxsize=None)
    def expr_ellipse() -> sy.Expr:
        """Generate a sympy expression for an ellipse."""
        return r / sy.sqrt(1 + (sy.tan(theta_0)**2) * (sy.cos(alpha)**2))

class NumericalFunctions:
    @staticmethod
    def calc_q(periastron: float, bh_mass: float, tol: float = 1e-3) -> float:
        """Convert Periastron distance P to the variable Q."""
        return np.sqrt((periastron - 2*bh_mass) * (periastron + 6*bh_mass))

    @staticmethod
    def calc_b_from_periastron(periastron: float, bh_mass: float, tol: float = 1e-5) -> float:
        """Get impact parameter b from Periastron distance P."""
        return np.sqrt((periastron**3) / (periastron - 2*bh_mass))

    @staticmethod
    def k(periastron: float, bh_mass: float) -> float:
        """Calculate modulus of elliptic integral."""
        q = NumericalFunctions.calc_q(periastron, bh_mass)
        return np.sqrt((q - periastron + 6 * bh_mass) / (2 * q))

    @staticmethod
    def k2(periastron: float, bh_mass: float, tol: float = 1e-6) -> float:
        """Calculate the squared modulus of elliptic integral."""
        q = NumericalFunctions.calc_q(periastron, bh_mass)
        return (q - periastron + 6 * bh_mass) / (2 * q)

    @staticmethod
    def zeta_inf(periastron: float, bh_mass: float, tol: float = 1e-6) -> float:
        """Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)."""
        q = NumericalFunctions.calc_q(periastron, bh_mass)
        arg = (q - periastron + 2 * bh_mass) / (q - periastron + 6 * bh_mass)
        return np.arcsin(np.sqrt(arg))

    @staticmethod
    def zeta_r(periastron: float, r: float, bh_mass: float) -> float:
        """Calculate the elliptic integral argument Zeta_r for a given value of P and r."""
        q = NumericalFunctions.calc_q(periastron, bh_mass)
        a = (q - periastron + 2 * bh_mass + (4 * bh_mass * periastron) / r) / (q - periastron + (6 * bh_mass))
        return np.arcsin(np.sqrt(a))

class GeometricFunctions:
    @staticmethod
    def cos_gamma(_a: float, incl: float, tol: float = 1e-5) -> float:
        """Calculate the cos of the angle gamma."""
        if abs(incl) < tol:
            return 0
        return np.cos(_a) / np.sqrt(np.cos(_a) ** 2 + 1 / (np.tan(incl) ** 2))

    @staticmethod
    def cos_alpha(phi: float, incl: float) -> float:
        """Returns cos(angle) alpha in observer frame given angles phi (black hole frame) and inclination (black hole frame)."""
        return np.cos(phi) * np.cos(incl) / np.sqrt((1 - np.sin(incl) ** 2 * np.cos(phi) ** 2))

    @staticmethod
    def alpha(phi: float, incl: float) -> float:
        """Returns observer coordinate of photon given phi (BHF) and inclination (BHF)."""
        return np.arccos(GeometricFunctions.cos_alpha(phi, incl))

    @staticmethod
    def ellipse(r: float, a: float, incl: float) -> float:
        """Equation of an ellipse, reusing the definition of cos_gamma."""
        g = np.arccos(GeometricFunctions.cos_gamma(a, incl))
        return r * np.sin(g)

class PhysicalFunctions:
    @staticmethod
    def eq13(periastron: float, ir_radius: float, ir_angle: float, bh_mass: float, incl: float, n: int = 0, tol: float = 1e-6) -> float:
        """Relation between radius (where photon was emitted in accretion disk), a and P."""
        z_inf = NumericalFunctions.zeta_inf(periastron, bh_mass)
        q = NumericalFunctions.calc_q(periastron, bh_mass)
        m_ = NumericalFunctions.k2(periastron, bh_mass)
        ell_inf = sp.ellipkinc(z_inf, m_)
        g = np.arccos(GeometricFunctions.cos_gamma(ir_angle, incl))

        if n:  # higher order image
            ell_k = sp.ellipk(m_)
            ellips_arg = (g - 2. * n * np.pi) / (2. * np.sqrt(periastron / q)) - ell_inf + 2. * ell_k
        else:  # direct image
            ellips_arg = g / (2. * np.sqrt(periastron / q)) + ell_inf

        sn, cn, dn, ph = sp.ellipj(ellips_arg, m_)
        sn2 = sn * sn
        term1 = -(q - periastron + 2. * bh_mass) / (4. * bh_mass * periastron)
        term2 = ((q - periastron + 6. * bh_mass) / (4. * bh_mass * periastron)) * sn2

        return 1. - ir_radius * (term1 + term2)

    @staticmethod
    def redshift_factor(radius: float, angle: float, incl: float, bh_mass: float, b_: float) -> float:
        """Calculate the gravitational redshift factor (1 + z)."""
        return (1 + np.sqrt(bh_mass / radius**3) * b_ * np.sin(incl) * np.sin(angle)) / np.sqrt(1 - 3*bh_mass / radius)

    @staticmethod
    def flux_intrinsic(r_val: float, acc: float, bh_mass: float) -> float:
        """Calculate the intrinsic flux of the accretion disk."""
        r_ = r_val / bh_mass
        log_arg = ((np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / \
                  ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
        return (3. * bh_mass * acc / (8 * np.pi)) * (1 / ((r_ - 3) * r_val ** 2.5)) * \
               (np.sqrt(r_) - np.sqrt(6) + 3 ** -.5 * np.log10(log_arg))

    @staticmethod
    def flux_observed(r_val: float, acc: float, bh_mass: float, redshift_factor: float) -> float:
        """Calculate the observed flux."""
        return PhysicalFunctions.flux_intrinsic(r_val, acc, bh_mass) / redshift_factor**4

    @staticmethod
    def phi_inf(periastron: float, M: float) -> float:
        """Calculate phi_inf."""
        q = NumericalFunctions.calc_q(periastron, M)
        ksq = NumericalFunctions.k2(periastron, M)
        z_inf = NumericalFunctions.zeta_inf(periastron, M)
        return 2. * (np.sqrt(periastron / q)) * (sp.ellipk(ksq) - sp.ellipkinc(z_inf, ksq))

    @staticmethod
    def mu(periastron: float, bh_mass: float) -> float:
        """Calculate mu."""
        return float(2 * PhysicalFunctions.phi_inf(periastron, bh_mass) - np.pi)

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

class Optimization:
    @staticmethod
    def midpoint_method(func: Callable, args: Dict[str, Any], x: list, y: list, ind: int) -> Tuple[list, list, int]:
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

        return new_x, new_y, new_ind

    @staticmethod
    def improve_solutions_midpoint(func: Callable, args: Dict[str, Any], x: list, y: list, index_of_sign_change: int, iterations: int) -> float:
        """Improve solutions using the midpoint method."""
        new_x, new_y = x, y
        new_ind = index_of_sign_change
        for _ in range(iterations):
            new_x, new_y, new_ind = Optimization.midpoint_method(func=func, args=args, x=new_x, y=new_y, ind=new_ind)
        updated_periastron = new_x[new_ind]
        return updated_periastron

    @staticmethod
    def calc_periastron(_r: float, incl: float, _alpha: float, bh_mass: float, midpoint_iterations: int = 100, 
                        plot_inbetween: bool = False, n: int = 0, min_periastron: float = 1., initial_guesses: int = 20) -> Optional[float]:
        """Calculate the periastron for given parameters."""
        periastron_range = list(np.linspace(min_periastron, 2. * _r, initial_guesses))
        y_ = [PhysicalFunctions.eq13(P_value, _r, _alpha, bh_mass, incl, n) for P_value in periastron_range]
        ind = np.where(np.diff(np.sign(y_)))[0]
        periastron_solution = periastron_range[ind[0]] if len(ind) > 0 else None

        if (periastron_solution is not None) and (not np.isnan(periastron_solution)):
            args_eq13 = {"ir_radius": _r, "ir_angle": _alpha, "bh_mass": bh_mass, "incl": incl, "n": n}
            periastron_solution = Optimization.improve_solutions_midpoint(
                func=PhysicalFunctions.eq13, args=args_eq13,
                x=periastron_range, y=y_, index_of_sign_change=ind[0] if len(ind) > 0 else 0,
                iterations=midpoint_iterations
            )
        return periastron_solution

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


# Main execution
if __name__ == "__main__":
    # Example usage
    M = 1
    solver_params = {'initial_guesses': 10, 'midpoint_iterations': 10, 'plot_inbetween': False, 'min_periastron': 3.01 * M}
    
    # Generate sample data
    alpha_vals = np.linspace(0, 2*np.pi, 1000)
    r_vals = np.arange(6, 30, 2)
    theta_0 = 80 * np.pi / 180
    n_vals = [0, 1]
    acc = 1e-8
    
    image_data = Simulation.generate_image_data(alpha_vals, r_vals, theta_0, n_vals, M, acc, solver_params)
    print(image_data.head())