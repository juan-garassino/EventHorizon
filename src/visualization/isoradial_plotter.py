# isoradial_plotter.py

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
from .base_plotter import BasePlotter
from ..core.isoradial_model import Isoradial
from typing import Iterable, Optional, Tuple
import logging
import cmocean.cm as ccm
import matplotlib
import matplotlib.transforms as mt
import numpy.typing as npt
import scipy.interpolate as si

from ..core import bh
from ..core.bh import Isoradial as DataIsoradial

logger = logging.getLogger(__name__)

plt.style.use("dark_background")
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16

class IsoradialPlotter(BasePlotter):
    def __init__(self, config):
        super().__init__(config)
        self.verbose = config.get('verbose', False)
        logger.info("🎨 Initializing IsoradialPlotter")

    def plot(self, isoradial: Isoradial, ax: Optional[plt.Axes] = None, 
             colornorm: Tuple[float, float] = (0, 1), alpha: float = 1.0):
        logger.info("🎨 Starting to plot isoradial")
        if self.verbose:
            logger.debug(f"📏 Isoradial radius: {isoradial.radius:.2f}")
        
        if ax is None:
            ax = self.ax
        
        label = f'Isoradial (radius={isoradial.radius:.2f})'
        if self.config['plot_params']['redshift']:
            logger.info("🌈 Plotting with redshift colors")
            line = self._colorline(ax, isoradial.X, isoradial.Y, 
                            z=[e - 1 for e in isoradial.redshift_factors],
                            cmap=plt.get_cmap('RdBu_r'), alpha=alpha)
            line.set_label(label)
        else:
            logger.info("🖌️ Plotting with solid color")
            line, = ax.plot(isoradial.X, isoradial.Y, color=self.config['plot_params']['line_color'],
                    alpha=alpha, linestyle=self.config['plot_params']['linestyle'], label=label)
        
        if self.config['plot_params']['legend']:
            logger.info("🏷️ Adding legend to plot")
            ax.legend(prop={'size': 16})
        
        if len(isoradial.X) and len(isoradial.Y):
            logger.info("🔍 Setting plot limits")
            mx = 1.1 * max(np.max(isoradial.X), np.max(isoradial.Y))
            ax.set_xlim([-mx, mx])
            ax.set_ylim([-mx, mx])
        
        logger.info("✅ Isoradial plotting complete")
        
        return ax
    
    def _make_segments(self, x_values, y_values):
        logger.info("🧩 Creating line segments")
        points = np.array([x_values, y_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if self.verbose:
            logger.debug("📊 Line segments created")
        return segments

    def _colorline(self, ax, x_values, y_values, z=None, cmap=plt.get_cmap('RdBu_r'), 
                   norm=plt.Normalize(0, 1), alpha=1.0, linewidth=3):
        logger.info("🎨 Creating color line")
        
        if z is None:
            z = np.linspace(0.0, 1.0, len(x_values))
    
        segments = self._make_segments(x_values, y_values)
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                  linewidth=linewidth, alpha=alpha)
        ax.add_collection(lc)
        
        logger.info("✅ Color line created and added to plot")
        
        return lc

    def plot_redshift(self, isoradial: Isoradial, fig: Optional[plt.Figure] = None, 
                      ax: Optional[plt.Axes] = None, show: bool = True):
        logger.info("📊 Starting to plot redshift")
        
        if fig is None:
            fig = self.fig
        if ax is None:
            ax = self.ax
        
        ax.plot(isoradial.angles, [z - 1 for z in isoradial.redshift_factors])
        ax.set_title(f"Redshift values for isoradial\nR={isoradial.radius} | M = {isoradial.mass}")
        ax.set_xlim([0, 2 * np.pi])
        
        if self.verbose:
            logger.debug("📏 X-axis limits set to [0, 2π]")
            logger.debug("📊 Redshift data plotted")
        
        if show:
            logger.info("🖼️ Displaying the plot")
            self.show_plot()
        
        logger.info("✅ Redshift plotting complete")

    def generate_isoradials(
        self,
        th0: float,
        radii: npt.NDArray[np.float64],
        image_orders: Iterable[int],
        color: Optional[Tuple[float, float, float, float]] = None,
        cmap: Optional[matplotlib.colors.LinearSegmentedColormap] = None,
    ) -> matplotlib.figure.Figure:
        """Generate argument svg of the isoradials."""
        if self.verbose:
            logger.info("🎨 Starting to generate isoradials")
            logger.info(f"📐 Observer inclination: {th0}")
            logger.info(f"🔢 Number of radii: {len(radii)}")
            logger.info(f"🖼️ Number of image orders: {len(image_orders)}")

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("S")
        ax.set_axis_off()

        alpha = np.linspace(0, 2 * np.pi, 1000)
        theta_0 = th0 * np.pi / 180

        if cmap is None:
            cmap = ccm.ice
            if self.verbose:
                logger.info("🎨 Using default colormap: ice")

        for image_order in sorted(image_orders)[::-1]:
            if self.verbose:
                logger.info(f"🔄 Processing image order: {image_order}")
            for radius in radii:
                if color is None:
                    linecolor = cmap((radius - np.min(radii)) / (np.max(radii) - np.min(radii)))
                else:
                    linecolor = color

                iso = DataIsoradial(
                    bh.reorient_alpha(alpha, image_order),
                    bh.impact_parameter(alpha, radius, theta_0, image_order=image_order, black_hole_mass=1),
                    radius,
                    theta_0,
                    image_order,
                )

                ax.plot(iso.alpha, iso.impact_parameters, color=linecolor)

        if self.verbose:
            logger.info("✅ Isoradials generation complete")

        return fig

    def generate_scatter_image(
        self,
        ax: Optional[matplotlib.axes.Axes],
        alpha: npt.NDArray[np.float64],
        radii: npt.NDArray[np.float64],
        th0: float,
        image_orders: Iterable[int],
        black_hole_mass: float,
        cmap: Optional[matplotlib.colors.LinearSegmentedColormap] = None,
    ) -> matplotlib.figure.Figure:
        """Generate an image of the black hole using argument scatter plot."""
        if self.verbose:
            logger.info("🌌 Starting to generate scatter image")
            logger.info(f"📐 Observer inclination: {th0}")
            logger.info(f"🔢 Number of alpha values: {len(alpha)}")
            logger.info(f"🔢 Number of radii: {len(radii)}")
            logger.info(f"🖼️ Number of image orders: {len(image_orders)}")
            logger.info(f"⚫ Black hole mass: {black_hole_mass}")

        theta_0 = th0 * np.pi / 180
        df = bh.generate_image_data(alpha, radii, theta_0, image_orders, black_hole_mass, {"max_steps": 3})

        if self.verbose:
            logger.info(f"📊 Generated image data shape: {df.shape}")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
        else:
            fig = ax.get_figure()

        if cmap is None:
            cmap = ccm.gray
            if self.verbose:
                logger.info("🎨 Using default colormap: gray")

        ax.set_theta_zero_location("S")
        ax.set_axis_off()
        for image_order in sorted(image_orders, reverse=True):
            if self.verbose:
                logger.info(f"🔄 Processing image order: {image_order}")
            df_n = df.loc[df["image_order"] == image_order]
            ax.scatter(df_n["alpha"], df_n["impact_parameters"], c=df_n["flux"], cmap=cmap)

        if self.verbose:
            logger.info("✅ Scatter image generation complete")

        return fig

    def generate_image(
        self,
        ax: Optional[matplotlib.axes.Axes],
        alpha: npt.NDArray[np.float64],
        radii: npt.NDArray[np.float64],
        th0: float,
        image_orders: Iterable[int],
        black_hole_mass: float,
        cmap: Optional[matplotlib.colors.LinearSegmentedColormap],
    ) -> matplotlib.figure.Figure:
        """Generate an image of the black hole."""
        if self.verbose:
            logger.info("🌌 Starting to generate black hole image")
            logger.info(f"📐 Observer inclination: {th0}")
            logger.info(f"🔢 Number of alpha values: {len(alpha)}")
            logger.info(f"🔢 Number of radii: {len(radii)}")
            logger.info(f"🖼️ Number of image orders: {len(image_orders)}")
            logger.info(f"⚫ Black hole mass: {black_hole_mass}")

        theta_0 = th0 * np.pi / 180
        df = bh.generate_image_data(alpha, radii, theta_0, image_orders, black_hole_mass, {"max_steps": 3})

        if self.verbose:
            logger.info(f"📊 Generated image data shape: {df.shape}")

        if ax is None:
            _, ax = plt.subplots(figsize=(30, 30))

        ax.set_axis_off()

        minx, maxx = df["x_values"].min(), df["x_values"].max()
        miny, maxy = df["y_values"].min(), df["y_values"].max()

        xx, yy = np.meshgrid(
            np.linspace(minx, maxx, 10000),
            np.linspace(miny, maxy, 10000),
        )

        if self.verbose:
            logger.info(f"📏 Mesh grid shape: {xx.shape}")

        fluxgrid = np.zeros(xx.shape, dtype=float)

        for image_order in image_orders:
            if self.verbose:
                logger.info(f"🔄 Processing image order: {image_order}")
            df_n = df.loc[df["image_order"] == image_order]
            fluxgrid_n = si.griddata(
                (df_n["x_values"], df_n["y_values"]), df_n["flux"], (xx, yy), method="linear"
            )
            fluxgrid[fluxgrid == 0] += fluxgrid_n[fluxgrid == 0]

        cmap = ccm.gray
        if self.verbose:
            logger.info("🎨 Using gray colormap")

        transform = mt.Affine2D().rotate_deg_around(0, 0, -90)
        ax.imshow(
            fluxgrid,
            interpolation="bilinear",
            cmap=cmap,
            aspect="equal",
            origin="lower",
            extent=[minx, maxx, miny, maxy],
            transform=transform + ax.transData,
        )

        edges = transform.transform(
            np.array([(minx, miny), (maxx, miny), (minx, maxy), (maxx, maxy)])
        )

        ax.set_xlim(edges[:, 0].min(), edges[:, 0].max())
        ax.set_ylim(edges[:, 1].min(), edges[:, 1].max())

        if self.verbose:
            logger.info("✅ Black hole image generation complete")

        return ax.get_figure()
