import numpy as np
import matplotlib.pyplot as plt
from src.core.black_hole_model import BlackHole
from src.core.isoradial_model import Isoradial
from src.core.isoredshift_model import Isoredshift
from src.visualization.black_hole_plotter import BlackHolePlotter
from src.visualization.isoradial_plotter import IsoradialPlotter
from src.visualization.isoredshift_plotter import IsoredshiftPlotter

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors

# Set up parameters
M = 1.0

params = {
    "isoradial_angular_parameters": {
        "start_angle": 0,  # Starting angle for isoradial calculations (in radians)
        "end_angle": np.pi,  # Ending angle for isoradial calculations (in radians)
        "angular_precision": 50,  # Number of points to calculate along the isoradial
        "mirror": True,  # Whether to mirror the calculated points for symmetry
    },
    "isoradial_solver_parameters": {
        "initial_guess_count": 12,  # Number of initial guesses for the root-finding algorithm
        "midpoint_iterations": 20,  # Number of iteration_count for the midpoint method
        "plot_inbetween": False,  # Whether to plot intermediate steps (for debugging)
        "minimum_periastron": 3.001,  # Minimum allowed periastron distance (in units of M)
        "use_ellipse": True,  # Whether to use elliptical approximation when necessary
    },
    "isoredshift_solver_parameters": {
        "initial_guess_count": 12,  # Number of initial guesses for isoredshift calculations
        "midpoint_iterations": 12,  # Number of iteration_count for the midpoint method in isoredshift calculations
        "times_inbetween": 2,  # Number of times to refine the solution between known points
        "retry_angular_precision": 15,  # Angular precision for retry attempts
        "minimum_periastron": 3.01,  # Minimum allowed periastron distance for isoredshift (in units of M)
        "use_ellipse": True,  # Whether to use elliptical approximation for isoredshift when necessary
        "retry_tip": 50,  # Number of retry attempts for improving the tip of the isoredshift curve
        "initial_radial_precision": 15,  # Initial number of radial points to calculate
        "plot_inbetween": False,  # Whether to plot intermediate steps in isoredshift calculations
    },
    "plot_params": {
        "plot_isoredshifts_inbetween": True,  # Whether to plot intermediate isoredshift lines
        "save_plot": True,  # Whether to save the generated plots
        "plot_ellipse": True,  # Whether to plot the elliptical approximation
        "plot_core": True,  # Whether to plot the black hole core (event horizon)
        "redshift": True,  # Whether to color the plot based on redshift values
        "linestyle": "dashed",  # Style of the plotted lines
        "linewidth": 0.2,  # Width of the plotted lines
        "key": "",  # Additional key for plot identification (if needed)
        "face_color": "black",  # Background color of the plot
        "line_color": "white",  # Color of the plotted lines
        "text_color": "white",  # Color of the text in the plot
        "calculate_alpha": 1.0,  # Opacity of the plotted lines
        "show_grid": False,  # Whether to show argument grid on the plot
        "legend": False,  # Whether to show argument legend on the plot
        "orig_background": False,  # Whether to use the original background (if applicable)
        "plot_disk_edges": True,  # Whether to plot the edges of the accretion disk
        "ax_lim": [-100, 100],  # Limits of the x_values and y_values axes
        "fig_size": (10, 8),  # Add this line
        "title": "Black Hole Visualization",  # Add this line
        "dpi": 300,  # Add this line
        "results_folder": "results",  # Folder to save the generated plots
    },
}

inclination = 75 * np.pi / 180

def plot_single_isoradial():
    isoradial = Isoradial(params, radius=30 * M, inclination=inclination, mass=M, order=0)
    isoradial.calculate()
    
    plotter = IsoradialPlotter(params)
    ax = plotter.plot(isoradial)
    ax.set_title("Single Isoradial")
    plotter.save_plot("single_isoradial.png")

def plot_isoradial_redshift():
    isoradial = Isoradial(params, radius=30 * M, inclination=inclination, mass=M, order=0)
    isoradial.calculate()
    
    plotter = IsoradialPlotter(params)
    ax = plotter.plot_redshift(isoradial, show=False)
    plotter.save_plot("isoradial_redshift.png")

def plot_blackhole_isoradials():
    blackhole = BlackHole(params, mass=M, inclination=inclination, accretion_rate=1e-8)
    
    direct_r = list(range(5, 101, 10))
    ghost_r = list(range(5, 101, 20))
    
    blackhole.calc_isoradials(direct_r, ghost_r)
    
    plotter = BlackHolePlotter(params)
    ax = plotter.plot_isoradials(blackhole, direct_r, ghost_r)
    plotter.save_plot("blackhole_isoradials.png")

def plot_blackhole_isoredshifts():
    blackhole = BlackHole(params, mass=M, inclination=inclination, accretion_rate=1e-8)
    
    redshifts = [-0.5, -0.35, -0.15, 0.0, 0.15, 0.25, 0.5, 0.75, 1.0]
    
    blackhole.calc_isoredshifts(redshifts)
    
    plotter = BlackHolePlotter(params)
    ax = plotter.plot_isoredshifts(blackhole, redshifts=redshifts)
    plotter.save_plot("blackhole_isoredshifts.png")

def plot_improved_isoredshift():
    blackhole = BlackHole(params, mass=M, inclination=85 * np.pi / 180, accretion_rate=1e-8)
    
    redshift = 0.5
    isoredshift = Isoredshift(params, blackhole.inclination, redshift, blackhole.mass)
    
    plotter = IsoredshiftPlotter(params)
    ax = plotter.plot_with_improvement(isoredshift)
    ax.set_title(f"Improved Isoredshift (z = {redshift})")
    plotter.save_plot("improved_isoredshift.png")

def main():
    print("Plotting single isoradial...")
    plot_single_isoradial()
    
    print("Plotting isoradial redshift...")
    plot_isoradial_redshift()
    
    print("Plotting black hole isoradials...")
    plot_blackhole_isoradials()
    
    print("Plotting black hole isoredshifts...")
    plot_blackhole_isoredshifts()
    
    print("Plotting improved isoredshift...")
    plot_improved_isoredshift()

if __name__ == "__main__":
    main()