import numpy as np

class GeometricFunctions:
    @staticmethod
    def cos_gamma(_a: float, inclination: float, tolerance: float = 1e-5) -> float:
        """Calculate the cos of the angle gamma."""
        if abs(inclination) < tolerance:
            return 0
        return np.cos(_a) / np.sqrt(np.cos(_a) ** 2 + 1 / (np.tan(inclination) ** 2))

    @staticmethod
    def cos_alpha(phi: float, inclination: float) -> float:
        """Returns cos(angle) alpha in observer frame given angles phi (black hole frame) and inclination (black hole frame)."""
        return np.cos(phi) * np.cos(inclination) / np.sqrt((1 - np.sin(inclination) ** 2 * np.cos(phi) ** 2))

    @staticmethod
    def alpha(phi: float, inclination: float) -> float:
        """Returns observer coordinate of photon given phi (BHF) and inclination (BHF)."""
        return np.arccos(GeometricFunctions.cos_alpha(phi, inclination))

    @staticmethod
    def ellipse(radius: float, argument: float, inclination: float) -> float:
        """Equation of an ellipse, reusing the definition of cos_gamma."""
        g = np.arccos(GeometricFunctions.cos_gamma(argument, inclination))
        return radius * np.sin(g)