# black_hole_model.py

from .base_model import BaseModel
from .disk_edge import DiskEdge
from .isoradial_model import Isoradial
from .isoredshift_model import Isoredshift
from typing import Dict, List, Optional
import numpy as np

class BlackHole(BaseModel):
    def __init__(self, config: dict, mass: float, inclination: float, accretion_rate: float):
        super().__init__(config)
        self.mass = mass
        self.inclination = inclination
        self.accretion_rate = accretion_rate
        self.disk_outer_edge = DiskEdge(config, 50*mass, inclination, mass, 0)
        self.disk_inner_edge = DiskEdge(config, 6*mass, inclination, mass, 0)
        self.disk_apparent_inner_edge = DiskEdge(config, 3*mass, inclination, mass, 0)
        self.isoradials: Dict[float, Dict[int, Isoradial]] = {}
        self.isoredshifts: Dict[float, Isoredshift] = {}
        if self.config.get('verbose', False):
            print(f"Initialized BlackHole with mass: {mass}, inclination: {inclination}, accretion_rate: {accretion_rate}")
            print(f"Disk edges - Outer: {50*mass}M, Inner: {6*mass}M, Apparent Inner: {3*mass}M")

    def add_isoradial(self, isoradial: Isoradial, radius: float, order: int):
        if radius not in self.isoradials:
            self.isoradials[radius] = {}
        self.isoradials[radius][order] = isoradial
        if self.config.get('verbose', False):
            print(f"Added isoradial with radius: {radius}M, order: {order}")

    def calc_isoradials(self, direct_r: List[float], ghost_r: List[float]):
        if self.config.get('verbose', False):
            print("Calculating isoradials...")
        for radius in sorted(ghost_r):
            isoradial = Isoradial(self.config, radius, self.inclination, self.mass, order=1)
            self.add_isoradial(isoradial, radius, 1)
        for radius in sorted(direct_r):
            isoradial = Isoradial(self.config, radius, self.inclination, self.mass, order=0)
            self.add_isoradial(isoradial, radius, 0)
        if self.config.get('verbose', False):
            print(f"Calculated {len(ghost_r)} ghost isoradials and {len(direct_r)} direct isoradials")

    def calc_isoredshifts(self, redshifts: Optional[List[float]] = None):
        redshifts = redshifts or [-.15, 0., .1, .20, .5]
        if self.config.get('verbose', False):
            print(f"Calculating isoredshifts for redshifts: {redshifts}")

        def get_dirty_isoradials(bh):
            isoradials = []
            radial_precision = bh.config["isoredshift_solver_parameters"]["initial_radial_precision"]
            for radius in np.linspace(bh.disk_inner_edge.radius, bh.disk_outer_edge.radius, radial_precision):
                isoradial = Isoradial(bh.config, radius, bh.inclination, bh.mass, order=0)
                isoradials.append(isoradial)
            if bh.config.get('verbose', False):
                print(f"Generated {len(isoradials)} dirty isoradials for isoredshift calculation")
            return isoradials

        dirty_isoradials = get_dirty_isoradials(self)
        for redshift in redshifts:
            if self.config.get('verbose', False):
                print(f"Processing isoredshift for redshift: {redshift}")
            dirty_ir_copy = dirty_isoradials.copy()
            iz = Isoredshift(self.config, self.inclination, redshift, self.mass, isoradials=dirty_ir_copy)
            iz.improve()
            self.isoredshifts[redshift] = iz
            if self.config.get('verbose', False):
                print(f"Completed isoredshift calculation for redshift: {redshift}")
        if self.config.get('verbose', False):
            print(f"Finished calculating {len(self.isoredshifts)} isoredshifts")
        return self.isoredshifts