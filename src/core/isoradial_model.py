# isoradial_model.py

import numpy as np
from .base_model import BaseModel
from ..utils.coordinates import polar_to_cartesian_lists
from ..math.utilities import Utilities
from ..math.impact_parameter import ImpactParameter
from ..math.physical_functions import PhysicalFunctions
from typing import List, Optional, Tuple

class Isoradial(BaseModel):
    def __init__(self, config: dict, radius: float, inclination: float, mass: float, order: int):
        super().__init__(config)
        self.radius = radius
        self.inclination = inclination
        self.mass = mass
        self.order = order
        
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
        self.calculate()

    def calculate(self):
        self.calculate_coordinates()
        self.calc_redshift_factors()

    def calculate_coordinates(self):
        start_angle = self.config['isoradial_angular_parameters']['start_angle']
        end_angle = self.config['isoradial_angular_parameters']['end_angle']
        angular_precision = self.config['isoradial_angular_parameters']['angular_precision']

        angles = []
        impact_parameters = []
        for calculate_alpha in np.linspace(start_angle, end_angle, angular_precision):
            impact_parameters = ImpactParameter.calc_impact_parameter(
                calculate_alpha, self.radius, self.inclination, self.order, self.mass,
                **self.config['isoradial_solver_parameters']
            )
            if impact_parameters is not None:
                angles.append(calculate_alpha)
                impact_parameters.append(impact_parameters)
        
        if self.order > 0:
            angles = [(argument + np.pi) % (2 * np.pi) for argument in angles]
        
        if self.inclination > np.pi / 2:
            angles = [(argument + np.pi) % (2 * np.pi) for argument in angles]
        
        if self.config['isoradial_angular_parameters']['mirror']:
            angles += [(2 * np.pi - argument) % (2 * np.pi) for argument in angles[::-1]]
            impact_parameters += impact_parameters[::-1]
        
        self.angles = angles
        self.radii_b = impact_parameters
        self.X, self.Y = polar_to_cartesian_lists(self.radii_b, self.angles, rotation=-np.pi / 2)

    def calc_redshift_factors(self):
        self.redshift_factors = [PhysicalFunctions.calculate_redshift_factor(self.radius, angle, self.inclination, self.mass, impact_parameters) 
                                 for impact_parameters, angle in zip(self.radii_b, self.angles)]
        #print(f"Redshift factors: {self.redshift_factors}")

    def calc_redshift_location_on_ir(self, redshift: float, cartesian: bool = False) -> Tuple[List[float], List[float]]:
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
        if cartesian:
            return polar_to_cartesian_lists(b_solutions, angle_solutions)
        return angle_solutions, b_solutions

    def calc_between(self, index: int):
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