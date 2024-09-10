import numpy as np

class NumericalFunctions:
    @staticmethod
    def validate_parameters(periastron: float, black_hole_mass: float, verbose: bool = False) -> bool:
        """Validate the input parameters."""
        if periastron <= 2 * black_hole_mass:
            if verbose:
                print(f"Invalid periastron={periastron} for black_hole_mass={black_hole_mass}. Must be greater than 2 * black_hole_mass.")
            return False
        if black_hole_mass <= 0:
            if verbose:
                print(f"Invalid black_hole_mass={black_hole_mass}. Must be greater than zero.")
            return False
        if verbose:
            print(f"Parameters valid: periastron={periastron}, black_hole_mass={black_hole_mass}")
        return True

    @staticmethod
    def calculate_q_parameter(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Convert Periastron distance P to the variable Q."""
        if not NumericalFunctions.validate_parameters(periastron, black_hole_mass, verbose):
            return np.nan
        term = (periastron - 2 * black_hole_mass) * (periastron + 6 * black_hole_mass)
        if verbose:
            print(f"calculate_q_parameter: periastron={periastron}, black_hole_mass={black_hole_mass}, term={term}")
        if term <= 0:
            if verbose:
                print(f"Invalid sqrt argument: term={term}")
            return np.nan
        result = np.sqrt(term)
        if verbose:
            print(f"calculate_q_parameter result: {result}")
        return result

    @staticmethod
    def calculate_impact_parameter(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Get impact parameter b from Periastron distance P."""
        if not NumericalFunctions.validate_parameters(periastron, black_hole_mass, verbose):
            return np.nan
        result = np.sqrt((periastron**3) / (periastron - 2 * black_hole_mass))
        if verbose:
            print(f"calculate_impact_parameter: periastron={periastron}, black_hole_mass={black_hole_mass}, result={result}")
        return result

    @staticmethod
    def calculate_elliptic_integral_modulus(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate modulus of elliptic integral."""
        q = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose)
        if verbose:
            print(f"calculate_elliptic_integral_modulus: periastron={periastron}, black_hole_mass={black_hole_mass}, q={q}")
        if np.isnan(q) or q == 0:
            if verbose:
                print("Invalid q value or division by zero")
            return np.nan
        result = np.sqrt((q - periastron + 6 * black_hole_mass) / (2 * q))
        if verbose:
            print(f"calculate_elliptic_integral_modulus result: {result}")
        return result

    @staticmethod
    def calculate_squared_elliptic_integral_modulus(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate the squared modulus of elliptic integral."""
        q = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose)
        if verbose:
            print(f"calculate_squared_elliptic_integral_modulus: periastron={periastron}, black_hole_mass={black_hole_mass}, q={q}")
        if np.isnan(q) or q == 0:
            if verbose:
                print("Invalid q value or division by zero")
            return np.nan
        result = (q - periastron + 6 * black_hole_mass) / (2 * q)
        if verbose:
            print(f"calculate_squared_elliptic_integral_modulus result: {result}")
        return result

    @staticmethod
    def calculate_zeta_infinity(periastron: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate Zeta_inf for elliptic integral F(Zeta_inf, calculate_elliptic_integral_modulus)."""
        q = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose)
        if verbose:
            print(f"calculate_zeta_infinity: periastron={periastron}, black_hole_mass={black_hole_mass}, q={q}")
        if np.isnan(q) or (q - periastron + 6 * black_hole_mass) == 0:
            if verbose:
                print("Invalid q value or division by zero")
            return np.nan
        arg = (q - periastron + 2 * black_hole_mass) / (q - periastron + 6 * black_hole_mass)
        if not (0 <= arg <= 1):
            if verbose:
                print(f"calculate_zeta_infinity: Invalid argument for asin={arg}")
            return np.nan
        result = np.arcsin(np.sqrt(arg))
        if verbose:
            print(f"calculate_zeta_infinity result: {result}")
        return result

    @staticmethod
    def calculate_zeta_radius(periastron: float, radius: float, black_hole_mass: float, verbose: bool = False) -> float:
        """Calculate the elliptic integral argument Zeta_r for argument given value of P and radius."""
        q = NumericalFunctions.calculate_q_parameter(periastron, black_hole_mass, verbose)
        if verbose:
            print(f"calculate_zeta_radius: periastron={periastron}, radius={radius}, black_hole_mass={black_hole_mass}, q={q}")
        if not NumericalFunctions.validate_parameters(periastron, black_hole_mass, verbose) or np.isnan(q):
            return np.nan
        denominator = q - periastron + 6 * black_hole_mass
        if denominator == 0:
            if verbose:
                print("Division by zero")
            return np.nan
        argument = (q - periastron + 2 * black_hole_mass + (4 * black_hole_mass * periastron) / radius) / denominator
        if not (0 <= argument <= 1):
            if verbose:
                print(f"calculate_zeta_radius: Invalid argument for asin={argument}")
            return np.nan
        result = np.arcsin(np.sqrt(argument))
        if verbose:
            print(f"calculate_zeta_radius result: {result}")
        return result