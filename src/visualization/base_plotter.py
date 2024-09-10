# base_plotter.py

import matplotlib.pyplot as plt
from typing import Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

class BasePlotter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verbose = self.config.get('verbose', False)
        logger.info("🎨 Initializing BasePlotter")
        if self.verbose:
            logger.info(f"🔑 Config keys: {', '.join(self.config.keys())}")
        self.setup_plot()

    def setup_plot(self):
        logger.info("🖼️ Setting up plot")
        self.fig, self.ax = plt.subplots(figsize=self.config['plot_params']['fig_size'])
        self.ax.set_facecolor(self.config['plot_params']['face_color'])
        if self.config['plot_params']['show_grid']:
            self.ax.grid(color='grey')
            self.ax.tick_params(which='both', labelcolor=self.config['plot_params']['text_color'], labelsize=15)
            if self.verbose:
                logger.info("📏 Grid enabled")
        else:
            self.ax.grid(False)
            if self.verbose:
                logger.info("📏 Grid disabled")
        self.ax.set_xlim(self.config['plot_params']["ax_lim"])
        self.ax.set_ylim(self.config['plot_params']["ax_lim"])
        if self.verbose:
            logger.info("✅ Plot setup complete")

    def close_plot(self):
        logger.info("🚪 Closing plot")

    def save_plot(self, filename: str):
        logger.info(f"💾 Saving plot: {filename}")
        dpi = self.config['plot_params'].get('dpi', 300)
        face_color = self.config['plot_params'].get('face_color', 'black')
        results_folder = self.config['plot_params'].get('results_folder', 'results')
        os.makedirs(results_folder, exist_ok=True)
        full_path = os.path.join(results_folder, filename)
        self.fig.savefig(full_path, dpi=dpi, facecolor=face_color)
        if self.verbose:
            logger.info(f"📁 Plot saved to: {full_path}")
            logger.info(f"🖼️ DPI: {dpi}, Face color: {face_color}")
        self.close_plot()

    def show_plot(self):
        logger.info("👀 Displaying plot")
        plt.show()