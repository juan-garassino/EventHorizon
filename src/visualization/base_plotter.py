# base_plotter.py

import matplotlib.pyplot as plt
from typing import Dict, Any

class BasePlotter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=self.config['fig_size'])
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_facecolor(self.config['face_color'])
        if self.config['show_grid']:
            self.ax.grid(color='grey')
            self.ax.tick_params(which='both', labelcolor=self.config['text_color'], labelsize=15)
        else:
            self.ax.grid(False)
        self.ax.set_xlim(self.config["ax_lim"])
        self.ax.set_ylim(self.config["ax_lim"])

    def save_plot(self, filename: str):
        self.fig.savefig(filename, dpi=self.config['dpi'], facecolor=self.config['face_color'])

    def show_plot(self):
        plt.show()