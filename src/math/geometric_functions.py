import numpy as np

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