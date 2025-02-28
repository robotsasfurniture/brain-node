import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple
import threading
import queue
from .sound_localization import LocalizationResult
from .logging_config import setup_logger

logger = setup_logger(__name__, log_file="logs/visualization.log")


class LocalizationVisualizer:
    """Real-time visualization of sound source localization results."""

    def __init__(self):
        """Initialize the visualizer with a plot window."""
        self.data_queue = queue.Queue()
        self.should_stop = False
        self.points: List[Tuple[float, float]] = []
        self.plot_lock = threading.Lock()

        # Initialize plot in the main thread
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.setup_plot()

    def setup_plot(self):
        """Set up the plot with proper scaling and labels."""
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.grid(True)
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.set_title("Sound Source Localization")

        # Plot robot position at center (5,5)
        self.ax.plot(5, 5, "bs", markersize=10, label="Robot")
        self.ax.legend()

        # Draw circle for reference
        circle = plt.Circle(
            (5, 5), 2, fill=False, linestyle="--", color="gray", alpha=0.5
        )
        self.ax.add_artist(circle)

        # Force plot to update
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_plot(self, result: LocalizationResult):
        """Add new localization result to the plot.

        Args:
            result: LocalizationResult containing position information
        """
        try:
            # Transform coordinates to plot space (centered at 5,5)
            x = result.x + 5
            y = result.y + 5

            with self.plot_lock:
                # Add new point
                self.points.append((x, y))
                # Keep only last 10 points
                if len(self.points) > 10:
                    self.points.pop(0)

                # Clear previous points
                for artist in self.ax.collections:
                    artist.remove()

                # Plot points with gradient color (older points more transparent)
                for i, (px, py) in enumerate(self.points):
                    alpha = (i + 1) / len(self.points)  # Newer points more opaque
                    self.ax.scatter(px, py, c="red", alpha=alpha)

                # Update plot
                try:
                    self.fig.canvas.draw_idle()
                    self.fig.canvas.flush_events()
                except Exception as e:
                    logger.error(f"Error updating plot display: {e}")

        except Exception as e:
            logger.error(f"Error updating plot: {e}")

    def close(self):
        """Clean up visualization resources."""
        try:
            plt.close(self.fig)
        except Exception as e:
            logger.error(f"Error closing plot: {e}")
