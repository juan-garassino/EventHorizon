import numpy as np

class NumericalFunctions:
    @staticmethod
    def validate_params(periastron: float, bh_mass: float, verbose: bool = False) -> bool:
        """Validate the input parameters."""
        if periastron <= 2 * bh_mass:
            if verbose:
                print(f"Invalid periastron={periastron} for bh_mass={bh_mass}. Must be greater than 2 * bh_mass.")
            return False
        if bh_mass <= 0:
            if verbose:
                print(f"Invalid bh_mass={bh_mass}. Must be greater than zero.")
            return False
        if verbose:
            print(f"Parameters valid: periastron={periastron}, bh_mass={bh_mass}")
        return True

    @staticmethod
    def calc_q(periastron: float, bh_mass: float, verbose: bool = False) -> float:
        """Convert Periastron distance P to the variable Q."""
        if not NumericalFunctions.validate_params(periastron, bh_mass, verbose):
            return np.nan
        term = (periastron - 2 * bh_mass) * (periastron + 6 * bh_mass)
        if verbose:
            print(f"calc_q: periastron={periastron}, bh_mass={bh_mass}, term={term}")
        if term <= 0:
            if verbose:
                print(f"Invalid sqrt argument: term={term}")
            return np.nan
        result = np.sqrt(term)
        if verbose:
            print(f"calc_q result: {result}")
        return result

    @staticmethod
    def calc_b_from_periastron(periastron: float, bh_mass: float, verbose: bool = False) -> float:
        """Get impact parameter b from Periastron distance P."""
        if not NumericalFunctions.validate_params(periastron, bh_mass, verbose):
            return np.nan
        result = np.sqrt((periastron**3) / (periastron - 2 * bh_mass))
        if verbose:
            print(f"calc_b_from_periastron: periastron={periastron}, bh_mass={bh_mass}, result={result}")
        return result

    @staticmethod
    def k(periastron: float, bh_mass: float, verbose: bool = False) -> float:
        """Calculate modulus of elliptic integral."""
        q = NumericalFunctions.calc_q(periastron, bh_mass, verbose)
        if verbose:
            print(f"k: periastron={periastron}, bh_mass={bh_mass}, q={q}")
        if np.isnan(q) or q == 0:
            if verbose:
                print("Invalid q value or division by zero")
            return np.nan
        result = np.sqrt((q - periastron + 6 * bh_mass) / (2 * q))
        if verbose:
            print(f"k result: {result}")
        return result

    @staticmethod
    def k2(periastron: float, bh_mass: float, verbose: bool = False) -> float:
        """Calculate the squared modulus of elliptic integral."""
        q = NumericalFunctions.calc_q(periastron, bh_mass, verbose)
        if verbose:
            print(f"k2: periastron={periastron}, bh_mass={bh_mass}, q={q}")
        if np.isnan(q) or q == 0:
            if verbose:
                print("Invalid q value or division by zero")
            return np.nan
        result = (q - periastron + 6 * bh_mass) / (2 * q)
        if verbose:
            print(f"k2 result: {result}")
        return result

    @staticmethod
    def zeta_inf(periastron: float, bh_mass: float, verbose: bool = False) -> float:
        """Calculate Zeta_inf for elliptic integral F(Zeta_inf, k)."""
        q = NumericalFunctions.calc_q(periastron, bh_mass, verbose)
        if verbose:
            print(f"zeta_inf: periastron={periastron}, bh_mass={bh_mass}, q={q}")
        if np.isnan(q) or (q - periastron + 6 * bh_mass) == 0:
            if verbose:
                print("Invalid q value or division by zero")
            return np.nan
        arg = (q - periastron + 2 * bh_mass) / (q - periastron + 6 * bh_mass)
        if not (0 <= arg <= 1):
            if verbose:
                print(f"zeta_inf: Invalid argument for asin={arg}")
            return np.nan
        result = np.arcsin(np.sqrt(arg))
        if verbose:
            print(f"zeta_inf result: {result}")
        return result

    @staticmethod
    def zeta_r(periastron: float, r: float, bh_mass: float, verbose: bool = False) -> float:
        """Calculate the elliptic integral argument Zeta_r for a given value of P and r."""
        q = NumericalFunctions.calc_q(periastron, bh_mass, verbose)
        if verbose:
            print(f"zeta_r: periastron={periastron}, r={r}, bh_mass={bh_mass}, q={q}")
        if not NumericalFunctions.validate_params(periastron, bh_mass, verbose) or np.isnan(q):
            return np.nan
        denom = q - periastron + 6 * bh_mass
        if denom == 0:
            if verbose:
                print("Division by zero")
            return np.nan
        a = (q - periastron + 2 * bh_mass + (4 * bh_mass * periastron) / r) / denom
        if not (0 <= a <= 1):
            if verbose:
                print(f"zeta_r: Invalid argument for asin={a}")
            return np.nan
        result = np.arcsin(np.sqrt(a))
        if verbose:
            print(f"zeta_r result: {result}")
        return result