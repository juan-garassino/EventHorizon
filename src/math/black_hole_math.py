import matplotlib.pyplot as plt
import numpy as np
from src.math.simulation import Simulation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to validate parameters
def validate_parameters(alpha_vals, radii, theta_0, image_orders, accretion_rate, verbose=False):
    logger.info("🔍 Validating parameters...")
    
    if not all(0 <= val <= 2*np.pi for val in alpha_vals):
        error_msg = "❌ Alpha values must be within the range [0, 2π]"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not all(val > 2 for val in radii):  # Assuming M=1, so radius > 2M
        error_msg = "❌ Radial values must be greater than 2M"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not (0 <= theta_0 <= np.pi):
        error_msg = "❌ Theta_0 must be within the range [0, π]"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not all(val in [0, 1] for val in image_orders):
        error_msg = "❌ image_order values must be either 0 or 1"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if accretion_rate <= 0:
        error_msg = "❌ Acc must be a positive number"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("✅ All parameters are valid!")

    if verbose:
        logger.info(f"📊 Validated {len(alpha_vals)} alpha values")
        logger.info(f"📏 Validated {len(radii)} radii")
        logger.info(f"🔢 Validated {len(image_orders)} image orders")

# Main execution
if __name__ == "__main__":
    verbose = True
    logger.info("🚀 Starting Black Hole Simulation")
    
    # Example usage
    M = 1
    solver_params = {'initial_guess_count': 10, 'midpoint_iterations': 10, 'plot_inbetween': False, 'minimum_periastron': 3.01 * M}

    # Generate sample data
    alpha_vals = np.linspace(0, 2*np.pi, 1000)
    radii = np.arange(6, 30, 2)
    theta_0 = 80 * np.pi / 180
    image_orders = [0, 1]
    accretion_rate = 1e-8

    logger.info("📊 Simulation parameters set")
    if verbose:
        logger.info(f"🔢 Number of alpha values: {len(alpha_vals)}")
        logger.info(f"📏 Radii range: {radii[0]} to {radii[-1]}")
        logger.info(f"🔄 Theta_0: {theta_0:.2f} radians")
        logger.info(f"🖼️ Image orders: {image_orders}")
        logger.info(f"💨 Accretion rate: {accretion_rate}")

    # Validate parameters
    try:
        validate_parameters(alpha_vals, radii, theta_0, image_orders, accretion_rate, verbose)
    except ValueError as e:
        logger.error(f"❌ Parameter validation error: {e}")
        exit(1)

    # Generate image data
    logger.info("🖥️ Generating image data...")
    image_data = Simulation.generate_image_data(alpha_vals, radii, theta_0, image_orders, M, accretion_rate, solver_params)
    logger.info("✅ Image data generated successfully")
    
    if verbose:
        logger.info(f"📊 Image data shape: {image_data.shape}")
        logger.info(f"🔑 Image data columns: {', '.join(image_data.columns)}")

    # Visualization
    logger.info("🎨 Creating visualization...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(image_data['x_values'], image_data['y_values'], c=image_data['flux'], cmap='viridis')
    plt.colorbar(scatter, label='Flux')
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.title('Black Hole Accretion Disk Simulation')
    logger.info("🖼️ Displaying the plot...")
    plt.show()
    
    logger.info("🏁 Simulation completed successfully!")