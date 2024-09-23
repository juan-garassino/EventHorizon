# black_hole_plotter.py

from .base_plotter import BasePlotter
from ..core.black_hole_model import BlackHole
from typing import Optional, List, Tuple
from ..core.isoradial_model import Isoradial
from ..core.isoredshift_model import Isoredshift
from ..visualization.isoradial_plotter import IsoradialPlotter
from ..visualization.isoredshift_plotter import IsoredshiftPlotter

class BlackHolePlotter(BasePlotter):
    def plot_photon_sphere(self, black_hole: BlackHole, c: str = '--', linewidth: float = 1.0):
        x, y = black_hole.disk_inner_edge.X, black_hole.disk_inner_edge.Y
        self.ax.plot(x, y, color=c, zorder=0, linewidth=linewidth)
        return self.ax

    def plot_apparent_inner_edge(self, black_hole: BlackHole, linestyle: str = '--', linewidth: float = 1.0):
        if hasattr(black_hole, 'disk_apparent_inner_edge'):
            x, y = black_hole.disk_apparent_inner_edge.X, black_hole.disk_apparent_inner_edge.Y
            self.ax.plot(x, y, zorder=0, linestyle=linestyle, linewidth=linewidth)
        else:
            print("Warning: Black hole does not have argument disk_apparent_inner_edge attribute.")
        return self.ax

    def plot_isoradials(self, black_hole: BlackHole, direct_r: List[float], ghost_r: List[float], show: bool = False, linewidth: float = 1.0):
        black_hole.calc_isoradials(direct_r, ghost_r)

        for radius in sorted(ghost_r):
            isoradial = black_hole.isoradials[radius][1]
            self.plot_isoradial(isoradial, color_range=(-1, 1), alpha=0.5, linewidth=linewidth)

        for radius in sorted(direct_r):
            isoradial = black_hole.isoradials[radius][0]  
            self.plot_isoradial(isoradial, color_range=(-1, 1), alpha=1.0, linewidth=linewidth)

        if self.config['plot_params']['plot_core']:
            self.plot_apparent_inner_edge(black_hole, '--', linewidth=linewidth)

        self.ax.set_title(f"Isoradials for M={black_hole.mass}", color=self.config['plot_params']['text_color'])
        
        if self.config['plot_params']['legend']:
            self.ax.legend(prop={'size': 16})
        else:
            self.ax.legend().remove()

        if self.config['plot_params']['save_plot']:
            filename = self.config['plot_params'].get('title', 'black_hole_plot').replace(' ', '_').replace('°', '')
            self.save_plot(filename)
        elif show:
            self.show_plot()
        else:
            self.close_plot()

        return self.fig, self.ax
        
    def plot_isoradial(self, isoradial: Isoradial, color_range: Tuple[float, float] = (0, 1), alpha: float = 1.0, linewidth: float = 1.0):
        isoradial_plotter = IsoradialPlotter(self.config)
        isoradial_plotter.plot(isoradial, ax=self.ax, colornorm=color_range, alpha=alpha, linewidth=linewidth)

    def plot_isoredshifts(self, black_hole: BlackHole, redshifts: Optional[List[float]] = None, plot_core: bool = False, linewidth: float = 1.0):
        if redshifts is None:
            redshifts = [-.2, -.15, 0., .15, .25, .5, .75, 1.]
        
        black_hole.calc_isoredshifts(redshifts=redshifts)

        for redshift, irz in black_hole.isoredshifts.items():
            isoredshift_plotter = IsoredshiftPlotter(self.config)
            isoredshift_plotter.plot_with_improvement(irz, ax=self.ax, linewidth=linewidth)
            
        if plot_core:
            self.plot_apparent_inner_edge(black_hole, linestyle='-', linewidth=linewidth)
            
        self.ax.set_title("Isoredshift lines for M={}".format(black_hole.mass))
        self.show_plot()
        return self.fig, self.ax
