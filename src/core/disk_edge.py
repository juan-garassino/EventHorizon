# disk_edge.py

import logging
from .base_model import BaseModel
from ..utils.coordinates import polar_to_cartesian_lists
from typing import List
from .isoradial_model import Isoradial
import numpy as np

logger = logging.getLogger(__name__)

class DiskEdge(BaseModel):
    def __init__(self, config: dict, radius: float, inclination: float, mass: float, order: int):
        super().__init__(config)
        self.radius = radius
        self.inclination = inclination
        self.mass = mass
        self.order = order
        self.X: List[float] = []
        self.Y: List[float] = []
        self.verbose = config.get('verbose', False)
        logger.info(f"🌟 Initializing DiskEdge with radius: {radius}, inclination: {inclination}, mass: {mass}, order: {order}")
        if self.verbose:
            logger.info(f"🔢 Initial X and Y lists are empty")
        self.calculate()

    def calculate(self):
        logger.info("🔄 Starting DiskEdge calculation")
        if self.verbose:
            logger.info("🔍 Creating Isoradial object")
        ir = Isoradial(self.config, self.radius, self.inclination, self.mass, self.order)
        if self.order == 0 and self.radius == 6 * self.mass:
            logger.info("🔧 Adjusting radii_b for special case: order 0 and radius 6M")
            if self.verbose:
                logger.info("📏 Scaling radii_b by factor of 0.99")
            ir.radii_b = [0.99 * impact_parameters for impact_parameters in ir.radii_b]
        logger.info("🔄 Converting polar coordinates to cartesian")
        self.X, self.Y = polar_to_cartesian_lists(ir.radii_b, ir.angles, rotation=-np.pi / 2)
        logger.info("✅ DiskEdge calculation completed")
        if self.verbose:
            logger.info("🔢 X and Y lists have been populated")