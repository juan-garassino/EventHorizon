import sympy as sy
import scipy.special as sp
import scipy.optimize as sopt
import numpy.typing as npt
import pandas as pd
import multiprocessing as mp
from typing import Callable, Union, Tuple, Optional, Dict, Any, Iterable
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
import warnings
from .simulation import Simulation

plt.style.use('fivethirtyeight')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # six fivethirtyeight themed colors

# Function to validate parameters
def validate_parameters(alpha_vals, r_vals, theta_0, n_vals, acc):
    if not all(0 <= val <= 2*np.pi for val in alpha_vals):
        raise ValueError("Alpha values must be within the range [0, 2π]")
    if not all(val > 2 for val in r_vals):  # Assuming M=1, so r > 2M
        raise ValueError("Radial values must be greater than 2M")
    if not (0 <= theta_0 <= np.pi):
        raise ValueError("Theta_0 must be within the range [0, π]")
    if not all(val in [0, 1] for val in n_vals):
        raise ValueError("n values must be either 0 or 1")
    if acc <= 0:
        raise ValueError("Acc must be a positive number")

# Main execution
if __name__ == "__main__":
    # Example usage
    M = 1
    solver_params = {'initial_guesses': 10, 'midpoint_iterations': 10, 'plot_inbetween': False, 'min_periastron': 3.01 * M}

    # Generate sample data
    alpha_vals = np.linspace(0, 2*np.pi, 1000)
    r_vals = np.arange(6, 30, 2)
    theta_0 = 80 * np.pi / 180
    n_vals = [0, 1]
    acc = 1e-8

    # Validate parameters
    try:
        validate_parameters(alpha_vals, r_vals, theta_0, n_vals, acc)
    except ValueError as e:
        print(f"Parameter validation error: {e}")
        exit(1)

    # Generate image data
    image_data = Simulation.generate_image_data(alpha_vals, r_vals, theta_0, n_vals, M, acc, solver_params)
    print(image_data.head())