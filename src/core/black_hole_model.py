# black_hole_model.py

import logging
from .base_model import BaseModel
from .disk_edge import DiskEdge
from .isoradial_model import Isoradial
from .isoredshift_model import Isoredshift
from typing import Dict, List, Optional, Union, Callable, Iterable
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

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
        logger.info(f"🕳️ Initialized BlackHole with mass: {mass}, inclination: {inclination}, accretion_rate: {accretion_rate}")
        if self.config.get('verbose', False):
            logger.info(f"💿 Disk edges - Outer: {50*mass}M, Inner: {6*mass}M, Apparent Inner: {3*mass}M")

    def add_isoradial(self, isoradial: Isoradial, radius: float, order: int):
        if radius not in self.isoradials:
            self.isoradials[radius] = {}
        self.isoradials[radius][order] = isoradial
        logger.info(f"➕ Added isoradial with radius: {radius}M, order: {order}")
        if self.config.get('verbose', False):
            logger.info(f"📊 Current isoradials structure: {len(self.isoradials)} radii")

    def calc_isoradials(self, direct_r: List[float], ghost_r: List[float]):
        logger.info("🧮 Calculating isoradials...")
        if self.config.get('verbose', False):
            logger.info(f"📏 Number of direct radii: {len(direct_r)}")
            logger.info(f"👻 Number of ghost radii: {len(ghost_r)}")
        for radius in sorted(ghost_r):
            isoradial = Isoradial(self.config, radius, self.inclination, self.mass, order=1)
            self.add_isoradial(isoradial, radius, 1)
        for radius in sorted(direct_r):
            isoradial = Isoradial(self.config, radius, self.inclination, self.mass, order=0)
            self.add_isoradial(isoradial, radius, 0)
        logger.info(f"✅ Isoradial calculations complete")
        if self.config.get('verbose', False):
            logger.info(f"🔢 Total number of isoradials: {sum(len(radii) for radii in self.isoradials.values())}")

    def calc_isoredshifts(self, redshifts: Optional[List[float]] = None):
        redshifts = redshifts or [-.15, 0., .1, .20, .5]
        logger.info(f"🌈 Calculating isoredshifts")
        if self.config.get('verbose', False):
            logger.info(f"🎨 Number of redshift values: {len(redshifts)}")

        def get_dirty_isoradials(bh):
            isoradials = []
            radial_precision = bh.config["isoredshift_solver_parameters"]["initial_radial_precision"]
            logger.info(f"🔍 Generating dirty isoradials")
            if bh.config.get('verbose', False):
                logger.info(f"🔧 Radial precision: {radial_precision}")
            for radius in np.linspace(bh.disk_inner_edge.radius, bh.disk_outer_edge.radius, radial_precision):
                isoradial = Isoradial(bh.config, radius, bh.inclination, bh.mass, order=0)
                isoradials.append(isoradial)
            logger.info(f"✅ Generated dirty isoradials")
            if bh.config.get('verbose', False):
                logger.info(f"📊 Number of dirty isoradials: {len(isoradials)}")
            return isoradials

        dirty_isoradials = get_dirty_isoradials(self)
        for redshift in redshifts:
            logger.info(f"🔄 Processing isoredshift")
            dirty_ir_copy = dirty_isoradials.copy()
            iz = Isoredshift(self.config, self.inclination, redshift, self.mass, isoradials=dirty_ir_copy)
            iz.improve()
            self.isoredshifts[redshift] = iz
            logger.info(f"✅ Completed isoredshift calculation")
        logger.info(f"🎉 Finished isoredshift calculations")
        if self.config.get('verbose', False):
            logger.info(f"📊 Number of calculated isoredshifts: {len(self.isoredshifts)}")
        return self.isoredshifts