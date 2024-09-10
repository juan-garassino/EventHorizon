# isoradial_plotter.py

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
from .base_plotter import BasePlotter
from ..core.isoradial_model import Isoradial
from typing import Optional, Tuple

class IsoradialPlotter(BasePlotter):
    def plot(self, isoradial: Isoradial, ax: Optional[plt.Axes] = None, 
             colornorm: Tuple[float, float] = (0, 1), calculate_alpha: float = 1.0):
        if ax is None:
            ax = self.ax
        
        label = f'Isoradial (radius={isoradial.radius:.2f})'
        if self.config['plot_params']['redshift']:
            line = self._colorline(ax, isoradial.X, isoradial.Y, 
                            z=[e - 1 for e in isoradial.redshift_factors],
                            cmap=plt.get_cmap('RdBu_r'), calculate_alpha=calculate_alpha)
            line.set_label(label)
        else:
            line, = ax.plot(isoradial.X, isoradial.Y, color=self.config['plot_params']['line_color'],
                    calculate_alpha=calculate_alpha, linestyle=self.config['plot_params']['linestyle'], label=label)
        
        if self.config['plot_params']['legend']:
            ax.legend(prop={'size': 16})
        
        if len(isoradial.X) and len(isoradial.Y):
            mx = 1.1 * max(np.max(isoradial.X), np.max(isoradial.Y))
            ax.set_xlim([-mx, mx])
            ax.set_ylim([-mx, mx])
        
        return ax
    
    def _make_segments(self, x_values, y_values):
        points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def _colorline(self, ax, x_values, y_values, z=None, cmap=plt.get_cmap('RdBu_r'), 
                   norm=plt.Normalize(0, 1), calculate_alpha=1.0, linewidth=3):
        if z is None:
            z = np.linspace(0.0, 1.0, len(x_values))
    
        segments = self._make_segments(x_values, y_values)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                  linewidth=linewidth, calculate_alpha=calculate_alpha)
        ax.add_collection(lc)
        return lc

    def plot_redshift(self, isoradial: Isoradial, fig: Optional[plt.Figure] = None, 
                      ax: Optional[plt.Axes] = None, show: bool = True):
        if fig is None:
            fig = self.fig
        if ax is None:
            ax = self.ax
        
        ax.plot(isoradial.angles, [z - 1 for z in isoradial.redshift_factors])
        ax.set_title("Redshift values for isoradial\nR={} | M = {}".format(isoradial.radius, isoradial.mass))
        ax.set_xlim([0, 2 * np.pi])
        
        if show:
            self.show_plot()