# isoredshift_plotter.py

import matplotlib.pyplot as plt
from .base_plotter import BasePlotter
from ..core.isoredshift_model import Isoredshift
from typing import Optional

class IsoredshiftPlotter(BasePlotter):
    def plot(self, isoredshift: Isoredshift, ax: Optional[plt.Axes] = None, color: str = 'white', linewidth: float = 1.0):
        if ax is None:
            ax = self.ax
        
        ax.plot(isoredshift.y, [-e for e in isoredshift.x], color=color, linewidth=linewidth)
        return ax

    def plot_with_improvement(self, isoredshift: Isoredshift, ax: Optional[plt.Axes] = None, 
                              max_tries: int = 10, color: str = 'white', linewidth: float = 1.0):
        if ax is None:
            ax = self.ax
        
        self.plot(isoredshift, ax, color, linewidth=linewidth)
        tries = 0
        while len(isoredshift.ir_radii_w_co) < 10 and tries < max_tries:
            isoredshift.improve_between_all_solutions_once()
            tries += 1
        self.plot(isoredshift, ax, color, linewidth=linewidth)
        return ax
