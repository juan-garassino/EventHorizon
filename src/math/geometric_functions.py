import numpy as np

class GeometricFunctions:
    @staticmethod
    def calculate_cos_gamma(azimuthal_angle: float, inclination: float, tolerance: float = 1e-5) -> float:
        """Calculate the cos of the angle gamma."""
        if abs(inclination) < tolerance:
            return 0
        return np.cos(azimuthal_angle) / np.sqrt(np.cos(azimuthal_angle) ** 2 + 1 / (np.tan(inclination) ** 2))

    @staticmethod
    def calculate_cos_alpha(phi_black_hole_frame: float, inclination: float) -> float:
        """Returns cos(angle) calculate_alpha in observer frame given angles phi_black_hole_frame (black hole frame) and inclination (black hole frame)."""
        return np.cos(phi_black_hole_frame) * np.cos(inclination) / np.sqrt((1 - np.sin(inclination) ** 2 * np.cos(phi_black_hole_frame) ** 2))

    @staticmethod
    def calculate_alpha(phi_black_hole_frame: float, inclination: float) -> float:
        """Returns observer coordinate of photon given phi_black_hole_frame (BHF) and inclination (BHF)."""
        return np.arccos(GeometricFunctions.calculate_cos_alpha(phi_black_hole_frame, inclination))

    @staticmethod
    def calculate_ellipse_radius(radius: float, argument: float, inclination: float) -> float:
        """Equation of an calculate_ellipse_radius, reusing the definition of calculate_cos_gamma."""
        gamma = np.arccos(GeometricFunctions.calculate_cos_gamma(argument, inclination))
        return radius * np.sin(gamma)