import os
import datetime
import core.blackhole as blackhole
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean.cm as ccm
import matplotlib.transforms as mt
import scipy.interpolate as si
import out

# Set the plotting style
plt.style.use('dark_background')

def create_blackhole_image(th0, alpha, r_vals, n_vals, m, results_folder):
    """
    Generate and save a black hole image.
    
    Parameters:
    - th0: Angle parameter.
    - alpha: Array of alpha values.
    - r_vals: Array of radius values.
    - n_vals: Array of n values.
    - m: Mass parameter.
    - results_folder: Directory to save the generated images.
    """
    fig = out.generate_image(None, alpha, r_vals, th0, n_vals, m, None)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(os.path.join(results_folder, f'blackhole_{timestamp}.png'), dpi=300)
    plt.close(fig)  # Close the figure to free memory

def generate_isoradials_image(th0, r_vals, n_vals, results_folder, color=None, cmap=None):
    """
    Generate and save isoradials images.
    
    Parameters:
    - th0: Angle parameter.
    - r_vals: Array of radius values.
    - n_vals: Array of n values.
    - results_folder: Directory to save the generated images.
    - color: Optional color for the isoradials.
    - cmap: Optional colormap for the isoradials.
    """
    fig = out.generate_isoradials(th0, r_vals, n_vals, color, cmap)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(os.path.join(results_folder, f'isoradials_{timestamp}.png'), dpi=300)
    plt.close(fig)

def generate_scatter_image(th0, alpha, r_vals, n_vals, m, results_folder, cmap=None):
    """
    Generate and save a scatter image of the black hole.
    
    Parameters:
    - th0: Angle parameter.
    - alpha: Array of alpha values.
    - r_vals: Array of radius values.
    - n_vals: Array of n values.
    - m: Mass parameter.
    - results_folder: Directory to save the generated images.
    - cmap: Optional colormap for the scatter image.
    """
    fig = out.generate_scatter_image(None, alpha, r_vals, th0, n_vals, m, cmap)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(os.path.join(results_folder, f'scatter_{timestamp}.png'), dpi=300)
    plt.close(fig)

def main():
    # Parameters
    th0 = 80
    alpha = np.linspace(0, 2 * np.pi, 1000)
    r_vals = np.arange(6, 30, 0.5)
    n_vals = [0, 1]
    m = 1

    # Create results directory if it doesn't exist
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)

    # Generate and save images
    create_blackhole_image(th0, alpha, r_vals, n_vals, m, results_folder)
    generate_isoradials_image(th0, r_vals, n_vals, results_folder)
    generate_scatter_image(th0, alpha, r_vals, n_vals, m, results_folder)

if __name__ == "__main__":
    main()
