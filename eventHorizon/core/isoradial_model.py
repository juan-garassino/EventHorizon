"""
Isoradial model for constant radius curves in black hole spacetime.

This module provides the data model for isoradial curves, which represent
the apparent shape of matter at constant radius in the accretion disk
as seen by a distant observer.
"""
import ast
import configparser
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base model with physical parameters for black hole simulations."""

    def __init__(
        self,
        mass: float = 1.0,
        inclination: float = 80.0,
        accretion_rate: float = 1e-8,
        config: Optional[Dict[str, Any]] = None
    ):
        self.mass = mass
        self.inclination_deg = inclination
        self.inclination_rad = inclination * np.pi / 180
        self.accretion_rate = accretion_rate
        self.config = config if config is not None else {}
        # Derived parameters
        self.critical_impact_parameter = 3 * np.sqrt(3) * mass
        self.schwarzschild_radius = 2 * mass
        self.isco_radius = 6 * mass

    @staticmethod
    def read_parameters(config_file: str) -> Dict[str, Dict[str, Any]]:
        config = configparser.ConfigParser(inline_comment_prefixes='#')
        config.read(config_file)
        return {section: {key: _safe_literal(val) for key, val in config[section].items()}
                for section in config.sections()}

    def get_configuration(self) -> Dict[str, Any]:
        return {
            'mass': self.mass, 'inclination_deg': self.inclination_deg,
            'inclination_rad': self.inclination_rad, 'accretion_rate': self.accretion_rate,
            'critical_impact_parameter': self.critical_impact_parameter,
            'schwarzschild_radius': self.schwarzschild_radius,
            'isco_radius': self.isco_radius, 'config': self.config.copy()
        }

    def validate_configuration(self) -> Dict[str, bool]:
        return {
            'mass_positive': self.mass > 0,
            'inclination_valid': 0 <= self.inclination_deg <= 180,
            'accretion_rate_positive': self.accretion_rate > 0
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'mass': self.mass, 'inclination_deg': self.inclination_deg,
            'accretion_rate': self.accretion_rate,
            'critical_impact_parameter': self.critical_impact_parameter,
            'schwarzschild_radius': self.schwarzschild_radius,
            'isco_radius': self.isco_radius
        }

    @abstractmethod
    def calculate_impact_parameter(self, *args, **kwargs):
        pass


class Isoradial(BaseModel):
    """Model for isoradial curves (constant radius curves in the accretion disk)."""
    
    def __init__(
        self,
        mass: float = 1.0,
        inclination: float = 80.0,
        accretion_rate: float = 1e-8,
        radius: float = 10.0,
        image_order: int = 0,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize isoradial model.
        
        Parameters
        ----------
        mass : float
            Black hole mass
        inclination : float
            Observer inclination angle in degrees
        accretion_rate : float
            Accretion rate
        radius : float
            Radius of the isoradial curve
        image_order : int
            Image order (0=direct, 1=ghost, etc.)
        config : Optional[Dict[str, Any]]
            Additional configuration
        """
        super().__init__(mass, inclination, accretion_rate, config)
        self.radius = radius
        self.image_order = image_order
        
        # Calculated data
        self.angles: List[float] = []
        self.impact_parameters: List[float] = []
        self.redshift_factors: List[float] = []
        self.cartesian_x: List[float] = []
        self.cartesian_y: List[float] = []
        
        # Initialize physics engine for backward compatibility
        self._physics_engine = None
    
    def _get_physics_engine(self):
        """Get physics engine instance (lazy initialization)."""
        if self._physics_engine is None:
            from ..math.geodesics import UnifiedGeodesics
            self._physics_engine = UnifiedGeodesics(mass=self.mass)
        return self._physics_engine
    
    def calculate_impact_parameter(self, radius: float = None, angle: float = 0.0, image_order: int = None) -> Optional[float]:
        """Calculate impact parameter using the new physics engine.
        
        This maintains backward compatibility while using the enhanced algorithms.
        """
        # Use instance values if not provided
        if radius is None:
            radius = self.radius
        if image_order is None:
            image_order = self.image_order
            
        # Convert inclination to radians
        inclination_rad = self.inclination_deg * np.pi / 180.0
        
        # Use the enhanced impact parameter calculation
        physics_engine = self._get_physics_engine()
        return physics_engine.calc_impact_parameter(
            radius, inclination_rad, angle, image_order
        )
    
    def calculate_coordinates(self, angular_precision: int = 100, **kwargs):
        """Calculate isoradial coordinates (angles and impact parameters)."""
        # Generate angles around the circle
        self.angles = np.linspace(0, 2*np.pi, angular_precision)
        self.impact_parameters = []
        
        # Calculate impact parameter for each angle
        for angle in self.angles:
            impact_param = self.calculate_impact_parameter(self.radius, angle, self.image_order)
            self.impact_parameters.append(impact_param if impact_param is not None else 0.0)
        
        # Convert to Cartesian coordinates
        self.convert_to_cartesian()
    
    def calculate_redshift_factors(self, **kwargs):
        """Calculate redshift factors along the isoradial curve."""
        if len(self.angles) == 0 or len(self.impact_parameters) == 0:
            self.calculate_coordinates(**kwargs)
        
        self.redshift_factors = []
        physics_engine = self._get_physics_engine()
        inclination_rad = self.inclination_deg * np.pi / 180.0
        
        for angle, impact_param in zip(self.angles, self.impact_parameters):
            if impact_param > 0:
                redshift = physics_engine.redshift_factor(
                    self.radius, angle, inclination_rad, impact_param
                )
                self.redshift_factors.append(redshift)
            else:
                self.redshift_factors.append(1.0)  # No redshift if no solution
    
    def convert_to_cartesian(self, rotation: float = -np.pi/2):
        """Convert polar coordinates to Cartesian for plotting."""
        if len(self.angles) == 0 or len(self.impact_parameters) == 0:
            return
            
        self.cartesian_x = []
        self.cartesian_y = []
        
        for angle, impact_param in zip(self.angles, self.impact_parameters):
            if impact_param > 0:
                # Apply rotation and convert to Cartesian
                rotated_angle = angle + rotation
                x = impact_param * np.cos(rotated_angle)
                y = impact_param * np.sin(rotated_angle)
                self.cartesian_x.append(x)
                self.cartesian_y.append(y)
            else:
                # No visible point
                self.cartesian_x.append(np.nan)
                self.cartesian_y.append(np.nan)
    
    def find_redshift_location(self, target_redshift: float, tolerance: float = 0.01, **kwargs) -> Tuple[List[float], List[float]]:
        """Find locations on the isoradial where redshift equals target value."""
        if len(self.redshift_factors) == 0:
            self.calculate_redshift_factors(**kwargs)
        
        matching_x = []
        matching_y = []
        
        for i, redshift in enumerate(self.redshift_factors):
            if abs(redshift - target_redshift) <= tolerance:
                if i < len(self.cartesian_x) and i < len(self.cartesian_y):
                    x, y = self.cartesian_x[i], self.cartesian_y[i]
                    if not (np.isnan(x) or np.isnan(y)):
                        matching_x.append(x)
                        matching_y.append(y)
        
        return matching_x, matching_y
    
    def interpolate_between_points(self, index: int, n_points: int = 10, **kwargs):
        """Interpolate additional points between existing points for higher precision."""
        if index >= len(self.angles) - 1:
            return
        
        # Get current and next angles
        angle1 = self.angles[index]
        angle2 = self.angles[index + 1]
        
        # Interpolate angles
        interp_angles = np.linspace(angle1, angle2, n_points + 2)[1:-1]  # Exclude endpoints
        
        # Calculate impact parameters for interpolated angles
        interp_impact_params = []
        for angle in interp_angles:
            impact_param = self.calculate_impact_parameter(self.radius, angle, self.image_order)
            interp_impact_params.append(impact_param if impact_param is not None else 0.0)
        
        # Insert interpolated points
        for i, (angle, impact_param) in enumerate(zip(interp_angles, interp_impact_params)):
            insert_index = index + 1 + i
            self.angles.insert(insert_index, angle)
            self.impact_parameters.insert(insert_index, impact_param)
        
        # Recalculate Cartesian coordinates
        self.convert_to_cartesian()
    
    def get_data_dict(self) -> Dict[str, Any]:
        """Get isoradial data as dictionary for plotting/export."""
        return {
            'radius': self.radius,
            'image_order': self.image_order,
            'angles': self.angles,
            'impact_parameters': self.impact_parameters,
            'redshift_factors': self.redshift_factors,
            'cartesian_x': self.cartesian_x,
            'cartesian_y': self.cartesian_y,
            'mass': self.mass,
            'inclination_deg': self.inclination_deg
        }


class IsoredshiftModel(BaseModel):
    """Model for isoredshift curves (constant redshift curves)."""
    
    def __init__(
        self,
        mass: float = 1.0,
        inclination: float = 80.0,
        accretion_rate: float = 1e-8,
        redshift_value: float = 0.0,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize isoredshift model.
        
        Parameters
        ----------
        mass : float
            Black hole mass
        inclination : float
            Observer inclination angle in degrees
        accretion_rate : float
            Accretion rate
        redshift_value : float
            Target redshift value for the curve
        config : Optional[Dict[str, Any]]
            Additional configuration
        """
        super().__init__(mass, inclination, accretion_rate, config)
        self.redshift_value = redshift_value
        
        # Calculated data
        self.coordinates: List[Tuple[float, float]] = []  # (angle, impact_parameter) pairs
        self.radii_with_solutions: Dict[float, List[List[float]]] = {}
        self.cartesian_x: List[float] = []
        self.cartesian_y: List[float] = []
    
    def calculate_impact_parameter(self, *args, **kwargs):
        """Calculate impact parameter (implementation from base class)."""
        pass
    
    def calculate_from_isoradials(self, isoradials: List[Isoradial], **kwargs):
        """Calculate isoredshift curve from a set of isoradial curves."""
        pass
    
    def improve_curve_precision(self, iterations: int = 5, **kwargs):
        """Improve curve precision through iterative refinement."""
        pass
    
    def detect_coordinate_jumps(self, threshold: float = 2.0) -> Optional[int]:
        """Detect jumps in coordinates that should not be connected."""
        pass
    
    def split_curve_on_jumps(self) -> List[List[Tuple[float, float]]]:
        """Split curve into segments at jump points."""
        pass
    
    def order_coordinates(self, **kwargs):
        """Order coordinates for proper curve visualization."""
        pass
    
    def get_data_dict(self) -> Dict[str, Any]:
        """Get isoredshift data as dictionary for plotting/export."""
        return {
            'redshift_value': self.redshift_value,
            'coordinates': self.coordinates,
            'cartesian_x': self.cartesian_x,
            'cartesian_y': self.cartesian_y,
            'mass': self.mass,
            'inclination_deg': self.inclination_deg
        }


class IsoGridModel(BaseModel):
    """Model for combined isoradial and isoredshift grid visualization."""
    
    def __init__(
        self,
        mass: float = 1.0,
        inclination: float = 80.0,
        accretion_rate: float = 1e-8,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize iso grid model."""
        super().__init__(mass, inclination, accretion_rate, config)
        
        self.isoradials: List[Isoradial] = []
        self.isoredshifts: List[IsoredshiftModel] = []
    
    def calculate_impact_parameter(self, *args, **kwargs):
        """Calculate impact parameter (implementation from base class)."""
        pass
    
    def generate_isoradial_grid(self, radii: List[float], **kwargs):
        """Generate a grid of isoradial curves."""
        pass
    
    def generate_isoredshift_grid(self, redshift_values: List[float], **kwargs):
        """Generate a grid of isoredshift curves."""
        pass
    
    def create_coordinate_system(self, radii: List[float], redshift_values: List[float], **kwargs):
        """Create a complete coordinate system with both iso curves."""
        pass
    
    def get_grid_data(self) -> Dict[str, Any]:
        """Get complete grid data for visualization."""
        return {
            'isoradials': [iso.get_data_dict() for iso in self.isoradials],
            'isoredshifts': [iso.get_data_dict() for iso in self.isoredshifts],
            'mass': self.mass,
            'inclination_deg': self.inclination_deg
        }


class PartialIsoradialModel(BaseModel):
    """Model for partial isoradial segments represented as particles."""
    
    def __init__(
        self,
        mass: float = 1.0,
        inclination: float = 80.0,
        accretion_rate: float = 1e-8,
        radius: float = 10.0,
        n_segments: int = 8,
        segment_length: float = 0.3,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize partial isoradial model.
        
        Parameters
        ----------
        mass : float
            Black hole mass
        inclination : float
            Observer inclination angle in degrees
        accretion_rate : float
            Accretion rate
        radius : float
            Radius of the isoradial
        n_segments : int
            Number of partial segments around the radius
        segment_length : float
            Length of each segment as fraction of full circle (0-1)
        config : Optional[Dict[str, Any]]
            Additional configuration
        """
        super().__init__(mass, inclination, accretion_rate, config)
        self.radius = radius
        self.n_segments = n_segments
        self.segment_length = segment_length
        
        # Calculated data
        self.segment_particles: List[List[Tuple[float, float]]] = []  # Particles for each segment
        self.particle_velocities: List[List[Tuple[float, float]]] = []  # Velocity vectors
        self.doppler_shifts: List[List[float]] = []  # Doppler shift values
        self.particle_colors: List[List[Tuple[float, float, float]]] = []  # RGB colors
    
    def calculate_impact_parameter(self, *args, **kwargs):
        """Calculate impact parameter (implementation from base class)."""
        pass
    
    def generate_segment_particles(self, particles_per_segment: int = 10, **kwargs):
        """Generate particles for each partial segment."""
        pass
    
    def calculate_orbital_velocities(self, rotation_direction: str = 'clockwise', **kwargs):
        """Calculate orbital velocities for Doppler shift calculation."""
        pass
    
    def apply_doppler_coloring(self, rotation_direction: str = 'clockwise', **kwargs):
        """Apply blue/red Doppler shift coloring to particles."""
        pass
    
    def calculate_particle_density_by_speed(self, speed_profile: str = 'keplerian', **kwargs):
        """Calculate particle density based on orbital speed."""
        pass
    
    def get_segment_data(self, segment_index: int) -> Dict[str, Any]:
        """Get data for a specific segment."""
        return {
            'segment_index': segment_index,
            'radius': self.radius,
            'particles': self.segment_particles[segment_index] if segment_index < len(self.segment_particles) else [],
            'velocities': self.particle_velocities[segment_index] if segment_index < len(self.particle_velocities) else [],
            'doppler_shifts': self.doppler_shifts[segment_index] if segment_index < len(self.doppler_shifts) else [],
            'colors': self.particle_colors[segment_index] if segment_index < len(self.particle_colors) else []
        }


class MultiRadiusPartialIsoradialModel(BaseModel):
    """Model for partial isoradial segments at multiple radii."""
    
    def __init__(
        self,
        mass: float = 1.0,
        inclination: float = 80.0,
        accretion_rate: float = 1e-8,
        radii_list: List[float] = None,
        segments_per_radius: int = 8,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize multi-radius partial isoradial model."""
        super().__init__(mass, inclination, accretion_rate, config)
        self.radii_list = radii_list or [6.0, 10.0, 15.0, 20.0, 30.0]
        self.segments_per_radius = segments_per_radius
        
        # Collection of partial isoradial models
        self.partial_isoradials: List[PartialIsoradialModel] = []
    
    def calculate_impact_parameter(self, *args, **kwargs):
        """Calculate impact parameter (implementation from base class)."""
        pass
    
    def generate_all_partial_isoradials(self, **kwargs):
        """Generate partial isoradial models for all radii."""
        pass
    
    def apply_velocity_field_coloring(self, rotation_direction: str = 'clockwise', **kwargs):
        """Apply velocity field coloring to all partial isoradials."""
        pass
    
    def calculate_speed_based_density(self, **kwargs):
        """Calculate particle density based on orbital speed at each radius."""
        pass
    
    def get_all_particles_data(self) -> Dict[str, Any]:
        """Get combined particle data from all radii."""
        return {
            'radii': self.radii_list,
            'segments_per_radius': self.segments_per_radius,
            'partial_isoradials': [iso.get_segment_data(i) for iso in self.partial_isoradials for i in range(iso.n_segments)],
            'mass': self.mass,
            'inclination_deg': self.inclination_deg
        }


class VelocityFieldModel(BaseModel):
    """Model for velocity field visualization in the accretion disk."""
    
    def __init__(
        self,
        mass: float = 1.0,
        inclination: float = 80.0,
        accretion_rate: float = 1e-8,
        rotation_direction: str = 'clockwise',
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize velocity field model."""
        super().__init__(mass, inclination, accretion_rate, config)
        self.rotation_direction = rotation_direction
        
        # Velocity field data
        self.velocity_particles: List[Tuple[float, float]] = []  # Particle positions
        self.velocity_vectors: List[Tuple[float, float]] = []    # Velocity directions
        self.doppler_colors: List[Tuple[float, float, float]] = []  # Doppler shift colors
    
    def calculate_impact_parameter(self, *args, **kwargs):
        """Calculate impact parameter (implementation from base class)."""
        pass
    
    def generate_velocity_field_particles(self, radii_list: List[float], **kwargs):
        """Generate particles representing the velocity field."""
        pass
    
    def calculate_doppler_shifts(self, observer_angle: float = 0.0, **kwargs):
        """Calculate Doppler shifts for velocity field visualization."""
        pass
    
    def create_streamlines(self, **kwargs):
        """Create streamline representations of the velocity field."""
        pass
    
    def get_velocity_field_data(self) -> Dict[str, Any]:
        """Get complete velocity field data."""
        return {
            'rotation_direction': self.rotation_direction,
            'particles': self.velocity_particles,
            'velocities': self.velocity_vectors,
            'colors': self.doppler_colors,
            'mass': self.mass,
            'inclination_deg': self.inclination_deg
        }

# Names an ini value may reference (e.g. ``end_angle = np.pi``).
_SAFE_NAMES = {"np.pi": np.pi, "np.e": np.e, "pi": np.pi}
_BINOPS = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: _bounded_pow(a, b),
}


def _bounded_pow(a, b):
    # An unbounded `9 ** 9 ** 9` would hang; config values never need this.
    if abs(b) > 64:
        raise ValueError(f"exponent too large: {b!r}")
    return a ** b


def _safe_literal(text: str) -> Any:
    """Parse an ini value as a Python literal, never executing code.

    Replaces a former ``eval(val)``. Accepts literals (numbers, strings,
    tuples, lists, booleans, None), the constants in ``_SAFE_NAMES`` and
    + - * / ** arithmetic on them. Anything else (calls, imports, other
    attribute access) raises ``ValueError``.
    """
    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            items = [ev(e) for e in node.elts]
            return tuple(items) if isinstance(node, ast.Tuple) else items
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, (ast.Name, ast.Attribute)) and ast.unparse(node) in _SAFE_NAMES:
            return _SAFE_NAMES[ast.unparse(node)]
        raise ValueError(f"unsupported ini value: {text!r}")

    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"unsupported ini value: {text!r}") from exc
    return ev(tree.body)
