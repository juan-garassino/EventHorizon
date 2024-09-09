import numpy as np
import sympy as sp
import scipy.special as scp
import warnings
from .numerical_functions import NumericalFunctions
from .geometric_functions import GeometricFunctions

class PhysicalFunctions:
    @staticmethod
    def eq13(periastron: float, ir_radius: float, ir_angle: float, bh_mass: float, incl: float, n: int = 0, tol: float = 1e-6, verbose: bool = False) -> float:
        """Relation between radius (where photon was emitted in accretion disk), a and P."""
        if verbose:
            print(f"eq13 input: periastron: {periastron}, ir_radius: {ir_radius}, ir_angle: {ir_angle}, bh_mass: {bh_mass}, incl: {incl}, n: {n}")

        z_inf = NumericalFunctions.zeta_inf(periastron, bh_mass, verbose=verbose)
        q = NumericalFunctions.calc_q(periastron, bh_mass, verbose=verbose)
        m_ = NumericalFunctions.k2(periastron, bh_mass, verbose=verbose)

        # Check for invalid q
        if q <= 0:
            if verbose:
                print(f"Invalid q value: {q}")
            return np.nan

        # Use elliptic_f instead of ellipkinc
        ell_inf = scp.ellipkinc(z_inf, m_)
        g = np.arccos(GeometricFunctions.cos_gamma(ir_angle, incl))

        # Check for division by zero or invalid values
        sqrt_term = np.sqrt(periastron / q)
        if sqrt_term == 0:
            if verbose:
                print(f"Invalid sqrt_term value: {sqrt_term}")
            return np.nan

        if n:  # higher order image
            ell_k = scp.ellipk(m_)
            ellips_arg = (g - 2. * n * np.pi) / (2. * sqrt_term) - ell_inf + 2. * ell_k
        else:  # direct image
            ellips_arg = g / (2. * sqrt_term) + ell_inf

        sn, cn, dn, ph = scp.ellipj(ellips_arg, m_)
        sn2 = sn * sn

        term1 = -(q - periastron + 2. * bh_mass) / (4. * bh_mass * periastron)
        term2 = ((q - periastron + 6. * bh_mass) / (4. * bh_mass * periastron)) * sn2

        if verbose:
            print(f"Intermediate values: z_inf: {z_inf}, q: {q}, m_: {m_}, ell_inf: {ell_inf}, g: {g}")
            print(f"ellips_arg: {ellips_arg}, sn: {sn}, cn: {cn}, dn: {dn}, ph: {ph}")
            print(f"term1: {term1}, term2: {term2}")

        result = 1. - ir_radius * (term1 + term2)
        
        if verbose:
            print(f"eq13 result: {result}")

        # Check for NaN result
        if np.isnan(result) and verbose:
            print(f"Result is NaN: {result}")

        return result

    @staticmethod
    def redshift_factor(radius: float, angle: float, incl: float, bh_mass: float, b_: float) -> float:
        """Calculate the gravitational redshift factor (1 + z)."""
        min_radius = 3.01 * bh_mass  # Set a minimum radius slightly larger than 3*bh_mass
        if radius < min_radius:
            radius = min_radius
            warnings.warn(f"Radius adjusted to minimum allowed value: {min_radius}")
        
        numerator = 1 + np.sqrt(bh_mass / radius**3) * b_ * np.sin(incl) * np.sin(angle)
        denominator = np.sqrt(1 - 3*bh_mass / radius)
        return numerator / denominator

    @staticmethod
    def flux_intrinsic(r_val: float, acc: float, bh_mass: float, verbose: bool = False) -> float:
        """Calculate the intrinsic flux of the accretion disk."""
        r_ = r_val / bh_mass
        log_arg = ((np.sqrt(r_) + np.sqrt(3)) * (np.sqrt(6) - np.sqrt(3))) / \
                  ((np.sqrt(r_) - np.sqrt(3)) * (np.sqrt(6) + np.sqrt(3)))
        result = (3. * bh_mass * acc / (8 * np.pi)) * (1 / ((r_ - 3) * r_val ** 2.5)) * \
                 (np.sqrt(r_) - np.sqrt(6) + 3 ** -.5 * np.log10(log_arg))
        if verbose:
            print(f"flux_intrinsic input: r_val: {r_val}, acc: {acc}, bh_mass: {bh_mass}")
            print(f"flux_intrinsic result: {result}")
        return result

    @staticmethod
    def flux_observed(r_val: float, acc: float, bh_mass: float, redshift_factor: float, verbose: bool = False) -> float:
        """Calculate the observed flux."""
        result = PhysicalFunctions.flux_intrinsic(r_val, acc, bh_mass, verbose) / redshift_factor**4
        if verbose:
            print(f"flux_observed input: r_val: {r_val}, acc: {acc}, bh_mass: {bh_mass}, redshift_factor: {redshift_factor}")
            print(f"flux_observed result: {result}")
        return result

    @staticmethod
    def phi_inf(periastron: float, M: float, verbose: bool = False) -> float:
        """Calculate phi_inf."""
        q = NumericalFunctions.calc_q(periastron, M, verbose)
        ksq = NumericalFunctions.k2(periastron, M, verbose)
        z_inf = NumericalFunctions.zeta_inf(periastron, M, verbose)
        result = 2. * (np.sqrt(periastron / q)) * (sp.ellipk(ksq) - sp.ellipkinc(z_inf, ksq))
        if verbose:
            print(f"phi_inf input: periastron: {periastron}, M: {M}")
            print(f"phi_inf result: {result}")
        return result

    @staticmethod
    def mu(periastron: float, bh_mass: float, verbose: bool = False) -> float:
        """Calculate mu."""
        result = float(2 * PhysicalFunctions.phi_inf(periastron, bh_mass, verbose) - np.pi)
        if verbose:
            print(f"mu input: periastron: {periastron}, bh_mass: {bh_mass}")
            print(f"mu result: {result}")
        return result