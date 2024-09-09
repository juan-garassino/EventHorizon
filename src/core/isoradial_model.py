# isoradial_model.py

import numpy as np
from .base_model import BaseModel
from ..utils.coordinates import polar_to_cartesian_lists
from ..math.black_hole_math import ImpactParameter, PhysicalFunctions #calc_impact_parameter, redshift_factor
from typing import List, Optional, Tuple

class Isoradial(BaseModel):
    def __init__(self, config: dict, radius: float, inclination: float, mass: float, order: int):
        super().__init__(config)
        self.radius = radius
        self.inclination = inclination
        self.mass = mass
        self.order = order
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
        for alpha in np.linspace(start_angle, end_angle, angular_precision):
            b = ImpactParameter.calc_impact_parameter(
                alpha, self.radius, self.inclination, self.order, self.mass,
                **self.config['isoradial_solver_parameters']
            )
            if b is not None:
                angles.append(alpha)
                impact_parameters.append(b)
        
        if self.order > 0:
            angles = [(a + np.pi) % (2 * np.pi) for a in angles]
        
        if self.inclination > np.pi / 2:
            angles = [(a + np.pi) % (2 * np.pi) for a in angles]
        
        if self.config['isoradial_angular_parameters']['mirror']:
            angles += [(2 * np.pi - a) % (2 * np.pi) for a in angles[::-1]]
            impact_parameters += impact_parameters[::-1]
        
        self.angles = angles
        self.radii_b = impact_parameters
        self.X, self.Y = polar_to_cartesian_lists(self.radii_b, self.angles, rotation=-np.pi / 2)

    def calc_redshift_factors(self):
        self.redshift_factors = [PhysicalFunctions.redshift_factor(self.radius, angle, self.inclination, self.mass, b) 
                                 for b, angle in zip(self.radii_b, self.angles)]
        print(f"Redshift factors: {self.redshift_factors}")

    def calc_redshift_location_on_ir(self, redshift: float, cartesian: bool = False) -> Tuple[List[float], List[float]]:
        diff = [redshift + 1 - z for z in self.redshift_factors]
        initial_guess_indices = np.where(np.diff(np.sign(diff)))[0]

        angle_solutions = []
        b_solutions = []
        if len(initial_guess_indices) > 0:
            for index in initial_guess_indices:
                new_ind = index
                for _ in range(self.config['isoradial_solver_parameters']["midpoint_iterations"]):
                    self.calc_between(new_ind)
                    diff_ = [redshift + 1 - z for z in self.redshift_factors[new_ind:new_ind + 3]]
                    start = np.where(np.diff(np.sign(diff_)))[0]
                    if len(start) > 0:
                        new_ind += start[0]
                    else:
                        break  # No sign change found, exit the loop
                angle_solutions.append(0.5 * (self.angles[new_ind] + self.angles[new_ind + 1]))
                b_solutions.append(0.5 * (self.radii_b[new_ind] + self.radii_b[new_ind + 1]))
        if cartesian:
            return polar_to_cartesian_lists(b_solutions, angle_solutions)
        return angle_solutions, b_solutions

    def calc_between(self, ind: int):
        mid_angle = 0.5 * (self.angles[ind] + self.angles[ind + 1])
        b = ImpactParameter.calc_impact_parameter(
            mid_angle,
            self.radius,
            self.inclination,
            self.order,
            self.mass,
            **self.config['isoradial_solver_parameters']
        )
        z = PhysicalFunctions.redshift_factor(self.radius, mid_angle, self.inclination, self.mass, b)
        self.radii_b.insert(ind + 1, b)
        self.angles.insert(ind + 1, mid_angle)
        self.redshift_factors.insert(ind + 1, z)