# isoredshift_plotter.py

import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import numpy as np
import numpy.typing as npt
import scipy.interpolate as si
import cmocean.cm as ccm
from .base_plotter import BasePlotter
from ..core.isoredshift_model import Isoredshift
from typing import Optional, Iterable, Tuple
import logging
import matplotlib

from ..core import bh
from ..core.bh import Isoradial as DataIsoradial

logger = logging.getLogger(__name__)

plt.style.use("dark_background")
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16

class IsoredshiftPlotter(BasePlotter):
    def __init__(self, config):
        super().__init__(config)
        self.verbose = config.get('verbose', False)
        logger.info("🎨 Initializing IsoredshiftPlotter")

    def plot(self, isoredshift: Isoredshift, ax: Optional[plt.Axes] = None, color: str = 'white'):
        logger.info("🖌️ Starting to plot isoredshift")
        
        if ax is None:
            ax = self.ax
        
        ax.plot(isoredshift.y_values, [-e for e in isoredshift.x_values], color=color)
        
        if self.verbose:
            logger.debug("📊 Isoredshift data plotted")
            logger.debug("🎨 Plot color: %s", color)
        
        logger.info("✅ Isoredshift plotting complete")
        
        return ax

    def plot_with_improvement(self, isoredshift: Isoredshift, ax: Optional[plt.Axes] = None, 
                              max_tries: int = 10, color: str = 'white'):
        logger.info("🔄 Starting plot with improvement")
        
        if ax is None:
            ax = self.ax
        
        self.plot(isoredshift, ax, color)
        tries = 0
        while len(isoredshift.ir_radii_w_co) < 10 and tries < max_tries:
            logger.info(f"🔁 Improvement iteration {tries + 1}")
            isoredshift.improve_between_all_solutions_once()
            tries += 1
            if self.verbose:
                logger.debug(f"📈 Completed iteration {tries}")
        
        logger.info(f"🏁 Improvement complete after {tries} iterations")
        
        self.plot(isoredshift, ax, color)
        
        if self.verbose:
            logger.debug("🖼️ Final plot rendered")
        
        logger.info("✅ Plot with improvement complete")
        
        return ax

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
                logger.info("🎨 Using default colormap: gray")

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
