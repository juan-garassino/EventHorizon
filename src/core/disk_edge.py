# disk_edge.py

from .base_model import BaseModel
from ..utils.coordinates import polar_to_cartesian_lists
from typing import List
from .isoradial_model import Isoradial
import numpy as np

class DiskEdge(BaseModel):
    def __init__(self, config: dict, radius: float, inclination: float, mass: float, order: int):
        super().__init__(config)
        self.radius = radius
        self.inclination = inclination
        self.mass = mass
        self.order = order
        self.X: List[float] = []
        self.Y: List[float] = []
        self.calculate()

    def calculate(self):
        ir = Isoradial(self.config, self.radius, self.inclination, self.mass, self.order)
        if self.order == 0 and self.radius == 6 * self.mass:
            ir.radii_b = [0.99 * impact_parameters for impact_parameters in ir.radii_b]
        self.X, self.Y = polar_to_cartesian_lists(ir.radii_b, ir.angles, rotation=-np.pi / 2)