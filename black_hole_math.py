import matplotlib.pyplot as plt
import numpy as np
from src.math.simulation import Simulation

# Function to validate parameters
def validate_parameters(alpha_vals, radii, theta_0, image_orders, accretion_rate):
    if not all(0 <= val <= 2*np.pi for val in alpha_vals):
        raise ValueError("Alpha values must be within the range [0, 2π]")
    if not all(val > 2 for val in radii):  # Assuming M=1, so radius > 2M
        raise ValueError("Radial values must be greater than 2M")
    if not (0 <= theta_0 <= np.pi):
        raise ValueError("Theta_0 must be within the range [0, π]")
    if not all(val in [0, 1] for val in image_orders):
        raise ValueError("image_order values must be either 0 or 1")
    if accretion_rate <= 0:
        raise ValueError("Acc must be argument positive number")

# Main execution
if __name__ == "__main__":
    # Example usage
    M = 1
    solver_params = {'initial_guess_count': 10, 'midpoint_iterations': 10, 'plot_inbetween': False, 'minimum_periastron': 3.01 * M}

    # Generate sample data
    alpha_vals = np.linspace(0, 2*np.pi, 1000)
    radii = np.arange(6, 30, 2)
    theta_0 = 80 * np.pi / 180
    image_orders = [0, 1]
    accretion_rate = 1e-8

    # Validate parameters
    try:
        validate_parameters(alpha_vals, radii, theta_0, image_orders, accretion_rate)
    except ValueError as e:
        print(f"Parameter validation error: {e}")
        exit(1)

    # Generate image data
    image_data = Simulation.generate_image_data(alpha_vals, radii, theta_0, image_orders, M, accretion_rate, solver_params)
    print(image_data.head())

    # TODO: Add visualization code here
    # For example:
    plt.figure(figsize=(10, 8))
    plt.scatter(image_data['x_values'], image_data['y_values'], c=image_data['flux'], cmap='viridis')
    plt.colorbar(label='Flux')
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.title('Black Hole Accretion Disk Simulation')
    plt.show()