# base_plotter.py

import matplotlib.pyplot as plt
from typing import Dict, Any
import os

class BasePlotter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_plot()

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=self.config['plot_params']['fig_size'])
        self.ax.set_facecolor(self.config['plot_params']['face_color'])
        if self.config['plot_params']['show_grid']:
            self.ax.grid(color='grey')
            self.ax.tick_params(which='both', labelcolor=self.config['plot_params']['text_color'], labelsize=15)
        else:
            self.ax.grid(False)
        self.ax.set_xlim(self.config['plot_params']["ax_lim"])
        self.ax.set_ylim(self.config['plot_params']["ax_lim"])

    def close_plot(self):
        plt.close(self.fig)

    def save_plot(self, filename: str):
        import os
        dpi = self.config['plot_params'].get('dpi', 300)
        face_color = self.config['plot_params'].get('face_color', 'black')
        results_folder = self.config['plot_params'].get('results_folder', 'results')
        os.makedirs(results_folder, exist_ok=True)
        full_path = os.path.join(results_folder, filename)
        self.fig.savefig(full_path, dpi=dpi, facecolor=face_color)
        print(f"Plot saved to {full_path}")
        self.close_plot()

    def show_plot(self):
        plt.show()