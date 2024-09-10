# isoradial_model.py

import numpy as np
import logging
from .base_model import BaseModel
from ..utils.coordinates import polar_to_cartesian_lists
from ..math.utilities import Utilities
from ..math.impact_parameter import ImpactParameter
from ..math.physical_functions import PhysicalFunctions
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

class Isoradial(BaseModel):
    def __init__(self, config: dict, radius: float, inclination: float, mass: float, order: int):
        super().__init__(config)
        self.radius = radius
        self.inclination = inclination
        self.mass = mass
        self.order = order
        self.verbose = config.get('verbose', False)
        
        # Initialize find_redshift_params
        self.find_redshift_params = {
            'force_redshift_solution': False,  # Default value
            'max_force_iter': 5  # Default value
        }
        
        self.angles: List[float] = []
        self.radii_b: List[float] = []
        self.redshift_factors: List[float] = []
        self.X: List[float] = []
        self.Y: List[float] = []
        
        logger.info(f"🌟 Initializing Isoradial model with radius: {radius}, inclination: {inclination}, mass: {mass}, order: {order}")
        
        self.calculate()

    def calculate(self):
        logger.info("🔄 Starting Isoradial calculations")
        self.calculate_coordinates()
        self.calc_redshift_factors()
        logger.info("✅ Isoradial calculations completed")

    def calculate_coordinates(self):
        logger.info("📐 Calculating coordinates")
        
        start_angle = self.config['isoradial_angular_parameters']['start_angle']
        end_angle = self.config['isoradial_angular_parameters']['end_angle']
        angular_precision = self.config['isoradial_angular_parameters']['angular_precision']

        self.angles = []
        self.radii_b = []
        for calculate_alpha in np.linspace(start_angle, end_angle, angular_precision):
            impact_parameter = ImpactParameter.calc_impact_parameter(
                calculate_alpha, self.radius, self.inclination, self.order, self.mass,
                **self.config['isoradial_solver_parameters']
            )
            if impact_parameter is not None:
                self.angles.append(calculate_alpha)
                self.radii_b.append(impact_parameter)
        
        if self.order > 0:
            self.angles = [(angle + np.pi) % (2 * np.pi) for angle in self.angles]
        
        if self.inclination > np.pi / 2:
            self.angles = [(angle + np.pi) % (2 * np.pi) for angle in self.angles]
        
        if self.config['isoradial_angular_parameters']['mirror']:
            self.angles += [(2 * np.pi - angle) % (2 * np.pi) for angle in self.angles[::-1]]
            self.radii_b += self.radii_b[::-1]
        
        self.X, self.Y = polar_to_cartesian_lists(self.radii_b, self.angles, rotation=-np.pi / 2)
        
        if self.verbose:
            logger.info(f"📊 Calculated coordinate points")
            logger.info(f"🔢 Number of angles calculated")
            logger.info(f"🔢 Number of radii calculated")

    def calc_redshift_factors(self):
        logger.info("🌈 Calculating redshift factors")
        
        self.redshift_factors = [PhysicalFunctions.calculate_redshift_factor(self.radius, angle, self.inclination, self.mass, impact_parameters) 
                                 for impact_parameters, angle in zip(self.radii_b, self.angles)]
        
        if self.verbose:
            logger.info(f"📊 Calculated redshift factors")
            logger.info(f"🔢 Number of redshift factors calculated")

    def calc_redshift_location_on_ir(self, redshift: float, cartesian: bool = False) -> Tuple[List[float], List[float]]:
        logger.info(f"🔍 Finding redshift location")
        
        diff = [redshift + 1 - z for z in self.redshift_factors]
        initial_guess_indices = np.where(np.diff(np.sign(diff)))[0]

        angle_solutions = []
        b_solutions = []
        if len(initial_guess_indices) > 0:
            for index in initial_guess_indices:
                new_index = index
                for _ in range(self.config['isoradial_solver_parameters']["midpoint_iterations"]):
                    self.calc_between(new_index)
                    diff_ = [redshift + 1 - z for z in self.redshift_factors[new_index:new_index + 3]]
                    start = np.where(np.diff(np.sign(diff_)))[0]
                    if len(start) > 0:
                        new_index += start[0]
                    else:
                        break  # No sign change found, exit the loop
                angle_solutions.append(0.5 * (self.angles[new_index] + self.angles[new_index + 1]))
                b_solutions.append(0.5 * (self.radii_b[new_index] + self.radii_b[new_index + 1]))
        
        if self.verbose:
            logger.info(f"✅ Redshift location search completed")
            logger.info(f"🔢 Number of solutions found")
        
        if cartesian:
            return polar_to_cartesian_lists(b_solutions, angle_solutions)
        return angle_solutions, b_solutions

    def calc_between(self, index: int):
        if self.verbose:
            logger.info(f"➗ Calculating midpoint")
        
        mid_angle = 0.5 * (self.angles[index] + self.angles[index + 1])
        impact_parameters = ImpactParameter.calc_impact_parameter(
            mid_angle,
            self.radius,
            self.inclination,
            self.order,
            self.mass,
            **self.config['isoradial_solver_parameters']
        )
        z = PhysicalFunctions.calculate_redshift_factor(self.radius, mid_angle, self.inclination, self.mass, impact_parameters)
        self.radii_b.insert(index + 1, impact_parameters)
        self.angles.insert(index + 1, mid_angle)
        self.redshift_factors.insert(index + 1, z)
        
        if self.verbose:
            logger.info(f"✅ Inserted new point")
            logger.info(f"🔢 Updated number of angles")
            logger.info(f"🔢 Updated number of radii")
            logger.info(f"🔢 Updated number of redshift factors")