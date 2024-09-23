# isoradial_plotter.py

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
from .base_plotter import BasePlotter
from ..core.isoradial_model import Isoradial
from typing import Optional, Tuple

class IsoradialPlotter(BasePlotter):
    def plot(self, isoradial: Isoradial, ax: Optional[plt.Axes] = None, 
             colornorm: Tuple[float, float] = (0, 1), alpha: float = 1.0, linewidth: float = 1.0):
        if ax is None:
            ax = self.ax
        
        label = f'Isoradial (radius={isoradial.radius:.2f})'
        if self.config['plot_params']['redshift']:
            line = self._colorline(ax, isoradial.X, isoradial.Y, 
                            z=[e - 1 for e in isoradial.redshift_factors],
                            cmap=plt.get_cmap('RdBu_r'), alpha=alpha, linewidth=linewidth)
            line.set_label(label)
        else:
            line, = ax.plot(isoradial.X, isoradial.Y, color=self.config['plot_params']['line_color'],
                            alpha=alpha, linestyle=self.config['plot_params']['linestyle'], label=label,
                            linewidth=linewidth)
        
        if self.config['plot_params']['legend']:
            ax.legend(prop={'size': 16})
        
        if len(isoradial.X) and len(isoradial.Y):
            mx = 1.1 * max(np.max(isoradial.X), np.max(isoradial.Y))
            ax.set_xlim([-mx, mx])
            ax.set_ylim([-mx, mx])
        
        return ax
    
    def _make_segments(self, x, y):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def _colorline(self, ax, x, y, z=None, cmap=plt.get_cmap('RdBu_r'), 
                   norm=plt.Normalize(0, 1), alpha=1.0, linewidth: float = 3.0):
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))
    
        segments = self._make_segments(x, y)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                  linewidth=linewidth, alpha=alpha)
        ax.add_collection(lc)
        return lc

    def plot_redshift(self, isoradial: Isoradial, fig: Optional[plt.Figure] = None, 
                      ax: Optional[plt.Axes] = None, show: bool = True, linewidth: float = 1.0):
        if fig is None:
            fig = self.fig
        if ax is None:
            ax = self.ax
        
        ax.plot(isoradial.angles, [z - 1 for z in isoradial.redshift_factors], linewidth=linewidth)
        ax.set_title("Redshift values for isoradial\nR={} | M = {}".format(isoradial.radius, isoradial.mass))
        ax.set_xlim([0, 2 * np.pi])
        
        if show:
            self.show_plot()
