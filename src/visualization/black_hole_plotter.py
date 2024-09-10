# black_hole_plotter.py

import logging
from .base_plotter import BasePlotter
from ..core.black_hole_model import BlackHole
from typing import Optional, List, Tuple, Iterable
from ..core.isoradial_model import Isoradial
from ..core.isoredshift_model import Isoredshift
from ..visualization.isoradial_plotter import IsoradialPlotter
from ..visualization.isoredshift_plotter import IsoredshiftPlotter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import numpy as np
import numpy.typing as npt
import scipy.interpolate as si
import cmocean.cm as ccm

from ..core import bh
from ..core.bh import Isoradial as DataIsoradial

logger = logging.getLogger(__name__)

plt.style.use("dark_background")
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16

class BlackHolePlotter(BasePlotter):
    def __init__(self, config):
        super().__init__(config)
        self.verbose = config.get('verbose', False)
        logger.info("🚀 Initializing BlackHolePlotter")

    def plot_photon_sphere(self, black_hole: BlackHole, c: str = '--'):
        logger.info("🌌 Plotting photon sphere")
        x_values, y_values = black_hole.disk_inner_edge.X, black_hole.disk_inner_edge.Y
        self.ax.plot(x_values, y_values, color=c, zorder=0)
        if self.verbose:
            logger.debug("📏 Photon sphere data prepared")
        logger.info("✅ Photon sphere plotted")
        return self.ax

    def plot_apparent_inner_edge(self, black_hole: BlackHole, linestyle: str = '--'):
        logger.info("🕳️ Plotting apparent inner edge")
        if hasattr(black_hole, 'disk_apparent_inner_edge'):
            x_values, y_values = black_hole.disk_apparent_inner_edge.X, black_hole.disk_apparent_inner_edge.Y
            self.ax.plot(x_values, y_values, zorder=0, linestyle=linestyle, linewidth=2. * self.config['plot_params']["linewidth"])
            if self.verbose:
                logger.debug("📏 Apparent inner edge data prepared")
            logger.info("✅ Apparent inner edge plotted")
        else:
            logger.warning("⚠️ Black hole does not have disk_apparent_inner_edge attribute")
        return self.ax

    def plot_isoradials(self, black_hole: BlackHole, direct_r: List[float], ghost_r: List[float], show: bool = False):
        logger.info("🌈 Starting isoradials plotting")
        if self.verbose:
            logger.debug(f"📊 Direct radii count: {len(direct_r)}")
            logger.debug(f"👻 Ghost radii count: {len(ghost_r)}")
        
        black_hole.calc_isoradials(direct_r, ghost_r)

        for radius in sorted(ghost_r):
            logger.info(f"👻 Plotting ghost isoradial")
            if self.verbose:
                logger.debug(f"📏 Ghost radius: {radius}")
            isoradial = black_hole.isoradials[radius][1]
            self.plot_isoradial(isoradial, color_range=(-1, 1), alpha=0.5)

        for radius in sorted(direct_r):
            logger.info(f"🎯 Plotting direct isoradial")
            if self.verbose:
                logger.debug(f"📏 Direct radius: {radius}")
            isoradial = black_hole.isoradials[radius][0]  
            self.plot_isoradial(isoradial, color_range=(-1, 1), alpha=1.0)

        if self.config['plot_params']['plot_core']:
            logger.info("🔵 Plotting core")
            self.plot_apparent_inner_edge(black_hole, '--')

        self.ax.set_title(f"Isoradials for M={black_hole.mass}", color=self.config['plot_params']['text_color'])
        
        if self.config['plot_params']['legend']:
            logger.info("🏷️ Adding legend")
            self.ax.legend(prop={'size': 16})
        else:
            logger.info("🚫 Removing legend")
            self.ax.legend().remove()

        if self.config['plot_params']['save_plot']:
            logger.info("💾 Saving plot")
            filename = self.config['plot_params'].get('title', 'black_hole_plot').replace(' ', '_').replace('°', '')
            self.save_plot(filename)
        elif show:
            logger.info("🖼️ Showing plot")
            self.show_plot()
        else:
            logger.info("🚪 Closing plot")
            self.close_plot()

        logger.info("✅ Isoradials plotting completed")
        return self.fig, self.ax
        
    def plot_isoradial(self, isoradial: Isoradial, color_range: Tuple[float, float] = (0, 1), alpha: float = 1.0):
        logger.info(f"🎨 Plotting individual isoradial")
        isoradial_plotter = IsoradialPlotter(self.config)
        isoradial_plotter.plot(isoradial, ax=self.ax, colornorm=color_range, alpha=alpha)
        if self.verbose:
            logger.debug("📊 Isoradial data applied to plot")
        logger.info("✅ Individual isoradial plotted")

    def plot_isoredshifts(self, black_hole: BlackHole, redshifts: Optional[List[float]] = None, plot_core: bool = False):
        logger.info("🌈 Starting isoredshifts plotting")
        
        if redshifts is None:
            redshifts = [-.2, -.15, 0., .15, .25, .5, .75, 1.]
        
        if self.verbose:
            logger.debug(f"🎨 Redshift values count: {len(redshifts)}")
        
        black_hole.calc_isoredshifts(redshifts=redshifts)

        for redshift, irz in black_hole.isoredshifts.items():
            logger.info(f"🔴 Plotting isoredshift")
            if self.verbose:
                logger.debug(f"📊 Redshift value: {redshift}")
            isoredshift_plotter = IsoredshiftPlotter(self.config)
            isoredshift_plotter.plot_with_improvement(irz, ax=self.ax)
            
        if plot_core:
            logger.info("🔵 Plotting core")
            self.plot_apparent_inner_edge(black_hole, linestyle='-')
            
        self.ax.set_title("Isoredshift lines for M={}".format(black_hole.mass))
        
        logger.info("🖼️ Showing plot")
        self.show_plot()
        
        logger.info("✅ Isoredshifts plotting completed")
        return self.fig, self.ax

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

                iso = Isoradial(
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
        cmap: Optional[matplotlib.colors.LinearSegmentedColormap] = None,
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

        if cmap is None:
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
