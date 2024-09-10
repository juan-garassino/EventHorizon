# """Generate output figures of black hole images."""
# from typing import Iterable, Optional, Tuple
# import logging

# import cmocean.cm as ccm
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.transforms as mt
# import numpy as np
# import numpy.typing as npt
# import scipy.interpolate as si

# import bh
# from data import Isoradial

# logger = logging.getLogger(__name__)

# plt.style.use("dark_background")
# plt.rcParams["axes.labelsize"] = 16
# plt.rcParams["axes.titlesize"] = 16
# plt.rcParams["xtick.labelsize"] = 16
# plt.rcParams["ytick.labelsize"] = 16


# class BlackHolePlotter:
#     def __init__(self, verbose: bool = False):
#         self.verbose = verbose
#         self.logger = logging.getLogger(__name__)

#     def generate_isoradials(
#         self,
#         th0: float,
#         radii: npt.NDArray[float],
#         image_orders: Iterable[int],
#         color: Optional[Tuple[float, float, float, float]] = None,
#         cmap: Optional[matplotlib.colors.LinearSegmentedColormap] = None,
#     ) -> matplotlib.figure.Figure:
#         """Generate argument svg of the isoradials."""
#         if self.verbose:
#             self.logger.info("🎨 Starting to generate isoradials")
#             self.logger.info(f"📐 Observer inclination: {th0}")
#             self.logger.info(f"🔢 Number of radii: {len(radii)}")
#             self.logger.info(f"🖼️ Number of image orders: {len(image_orders)}")

#         fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
#         ax.set_theta_zero_location("S")
#         ax.set_axis_off()

#         alpha = np.linspace(0, 2 * np.pi, 1000)
#         theta_0 = th0 * np.pi / 180

#         if cmap is None:
#             cmap = ccm.ice
#             if self.verbose:
#                 self.logger.info("🎨 Using default colormap: ice")

#         for image_order in sorted(image_orders)[::-1]:
#             if self.verbose:
#                 self.logger.info(f"🔄 Processing image order: {image_order}")
#             for radius in radii:
#                 if color is None:
#                     linecolor = cmap((radius - np.min(radii)) / (np.max(radii) - np.min(radii)))
#                 else:
#                     linecolor = color

#                 iso = Isoradial(
#                     bh.reorient_alpha(alpha, image_order),
#                     bh.impact_parameter(alpha, radius, theta_0, image_order=image_order, black_hole_mass=1),
#                     radius,
#                     theta_0,
#                     image_order,
#                 )

#                 ax.plot(iso.alpha, iso.impact_parameters, color=linecolor)

#         if self.verbose:
#             self.logger.info("✅ Isoradials generation complete")

#         return fig

#     def generate_scatter_image(
#         self,
#         ax: Optional[matplotlib.axes.Axes],
#         alpha: npt.NDArray[float],
#         radii: npt.NDArray[float],
#         th0: float,
#         image_orders: Iterable[int],
#         black_hole_mass: float,
#         cmap: Optional[matplotlib.colors.LinearSegmentedColormap] = None,
#     ) -> matplotlib.figure.Figure:
#         """Generate an image of the black hole using argument scatter plot."""
#         if self.verbose:
#             self.logger.info("🌌 Starting to generate scatter image")
#             self.logger.info(f"📐 Observer inclination: {th0}")
#             self.logger.info(f"🔢 Number of alpha values: {len(alpha)}")
#             self.logger.info(f"🔢 Number of radii: {len(radii)}")
#             self.logger.info(f"🖼️ Number of image orders: {len(image_orders)}")
#             self.logger.info(f"⚫ Black hole mass: {black_hole_mass}")

#         theta_0 = th0 * np.pi / 180
#         df = bh.generate_image_data(alpha, radii, theta_0, image_orders, black_hole_mass, {"max_steps": 3})

#         if self.verbose:
#             self.logger.info(f"📊 Generated image data shape: {df.shape}")

#         if ax is None:
#             fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
#         else:
#             fig = ax.get_figure()

#         if cmap is None:
#             cmap = ccm.gray
#             if self.verbose:
#                 self.logger.info("🎨 Using default colormap: gray")

#         ax.set_theta_zero_location("S")
#         ax.set_axis_off()
#         for image_order in sorted(image_orders, reverse=True):
#             if self.verbose:
#                 self.logger.info(f"🔄 Processing image order: {image_order}")
#             df_n = df.loc[df["image_order"] == image_order]
#             ax.scatter(df_n["alpha"], df_n["impact_parameters"], c=df_n["flux"], cmap=cmap)

#         if self.verbose:
#             self.logger.info("✅ Scatter image generation complete")

#         return fig

#     def generate_image(
#         self,
#         ax: Optional[matplotlib.axes.Axes],
#         alpha: npt.NDArray[float],
#         radii: npt.NDArray[float],
#         th0: float,
#         image_orders: Iterable[int],
#         black_hole_mass: float,
#         cmap: Optional[matplotlib.colors.LinearSegmentedColormap],
#     ) -> matplotlib.figure.Figure:
#         """Generate an image of the black hole."""
#         if self.verbose:
#             self.logger.info("🌌 Starting to generate black hole image")
#             self.logger.info(f"📐 Observer inclination: {th0}")
#             self.logger.info(f"🔢 Number of alpha values: {len(alpha)}")
#             self.logger.info(f"🔢 Number of radii: {len(radii)}")
#             self.logger.info(f"🖼️ Number of image orders: {len(image_orders)}")
#             self.logger.info(f"⚫ Black hole mass: {black_hole_mass}")

#         theta_0 = th0 * np.pi / 180
#         df = bh.generate_image_data(alpha, radii, theta_0, image_orders, black_hole_mass, {"max_steps": 3})

#         if self.verbose:
#             self.logger.info(f"📊 Generated image data shape: {df.shape}")

#         if ax is None:
#             _, ax = plt.subplots(figsize=(30, 30))

#         ax.set_axis_off()

#         minx, maxx = df["x_values"].min(), df["x_values"].max()
#         miny, maxy = df["y_values"].min(), df["y_values"].max()

#         xx, yy = np.meshgrid(
#             np.linspace(minx, maxx, 10000),
#             np.linspace(miny, maxy, 10000),
#         )

#         if self.verbose:
#             self.logger.info(f"📏 Mesh grid shape: {xx.shape}")

#         fluxgrid = np.zeros(xx.shape, dtype=float)

#         for image_order in image_orders:
#             if self.verbose:
#                 self.logger.info(f"🔄 Processing image order: {image_order}")
#             df_n = df.loc[df["image_order"] == image_order]
#             fluxgrid_n = si.griddata(
#                 (df_n["x_values"], df_n["y_values"]), df_n["flux"], (xx, yy), method="linear"
#             )
#             fluxgrid[fluxgrid == 0] += fluxgrid_n[fluxgrid == 0]

#         cmap = ccm.gray
#         if self.verbose:
#             self.logger.info("🎨 Using gray colormap")

#         transform = mt.Affine2D().rotate_deg_around(0, 0, -90)
#         ax.imshow(
#             fluxgrid,
#             interpolation="bilinear",
#             cmap=cmap,
#             aspect="equal",
#             origin="lower",
#             extent=[minx, maxx, miny, maxy],
#             transform=transform + ax.transData,
#         )

#         edges = transform.transform(
#             np.array([(minx, miny), (maxx, miny), (minx, maxy), (maxx, maxy)])
#         )

#         ax.set_xlim(edges[:, 0].min(), edges[:, 0].max())
#         ax.set_ylim(edges[:, 1].min(), edges[:, 1].max())

#         if self.verbose:
#             self.logger.info("✅ Black hole image generation complete")

#         return ax.get_figure()
