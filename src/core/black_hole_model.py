# black_hole_model.py

from .base_model import BaseModel
from .disk_edge import DiskEdge
from .isoradial_model import Isoradial
from .isoredshift_model import Isoredshift
from typing import Dict, List, Optional
import numpy as np

class BlackHole(BaseModel):
    def __init__(self, config: dict, mass: float, inclination: float, acc: float):
        super().__init__(config)
        self.mass = mass
        self.inclination = inclination
        self.acc = acc
        self.disk_outer_edge = DiskEdge(config, 50*mass, inclination, mass, 0)
        self.disk_inner_edge = DiskEdge(config, 6*mass, inclination, mass, 0)
        self.disk_apparent_inner_edge = DiskEdge(config, 3*mass, inclination, mass, 0)
        self.isoradials: Dict[float, Dict[int, Isoradial]] = {}
        self.isoredshifts: Dict[float, Isoredshift] = {}

    def add_isoradial(self, isoradial: Isoradial, radius: float, order: int):
        if radius not in self.isoradials:
            self.isoradials[radius] = {}
        self.isoradials[radius][order] = isoradial

    def calc_isoradials(self, direct_r: List[float], ghost_r: List[float]):
        for radius in sorted(ghost_r):
            isoradial = Isoradial(self.config, radius, self.inclination, self.mass, order=1)
            self.add_isoradial(isoradial, radius, 1)

        for radius in sorted(direct_r):
            isoradial = Isoradial(self.config, radius, self.inclination, self.mass, order=0)
            self.add_isoradial(isoradial, radius, 0)

    def calc_isoredshifts(self, redshifts: Optional[List[float]] = None):
        redshifts = redshifts or [-.15, 0., .1, .20, .5]

        def get_dirty_isoradials(bh):
            isoradials = []
            for radius in np.linspace(bh.disk_inner_edge.radius, bh.disk_outer_edge.radius,
                                      bh.config["isoredshift_solver_parameters"]["initial_radial_precision"]):
                isoradial = Isoradial(bh.config, radius, bh.inclination, bh.mass, order=0)
                isoradials.append(isoradial)
            return isoradials

        dirty_isoradials = get_dirty_isoradials(self)
        for redshift in redshifts:
            dirty_ir_copy = dirty_isoradials.copy()
            iz = Isoredshift(self.config, self.inclination, redshift, self.mass, isoradials=dirty_ir_copy)
            iz.improve()
            self.isoredshifts[redshift] = iz
        return self.isoredshifts