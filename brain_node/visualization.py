import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple
import threading
import queue
from .sound_localization import LocalizationResult
from .logging_config import setup_logger
import time
from matplotlib.animation import FuncAnimation

logger = setup_logger(__name__, log_file="logs/visualization.log")


class LocalizationVisualizer:
    """Real-time visualization of sound source localization results."""

    def __init__(self):
        """Initialize the visualizer with a plot window."""
        self.points: List[Tuple[float, float]] = []
        self.plot_lock = threading.Lock()
        self.update_queue = queue.Queue()
        self.should_stop = False

        # Initialize plot
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.scatter = None
        self.setup_plot()

        # Initialize animation
        self.anim = FuncAnimation(
            self.fig,
            self._animate,
            interval=100,  # Update every 100ms
            blit=True,
            cache_frame_data=False,
        )

        # Show the plot window without blocking
        plt.show(block=False)
        plt.pause(0.1)

    def setup_plot(self):
        """Set up the plot with proper scaling and labels."""
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.grid(True)
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.set_title("Sound Source Localization")

        # Plot robot position at center (5,5)
        self.robot_point = self.ax.plot(5, 5, "bs", markersize=10, label="Robot")[0]

        # Draw circle for reference
        self.circle = plt.Circle(
            (5, 5), 2, fill=False, linestyle="--", color="gray", alpha=0.5
        )
        self.ax.add_artist(self.circle)

        # Initialize empty scatter plot
        self.scatter = self.ax.scatter([], [], c="r", alpha=0.5, label="Sound Sources")

        # Add legend
        self.ax.legend()

    def _animate(self, frame):
        """Animation function called by FuncAnimation."""
        with self.plot_lock:
            if self.points:
                xs, ys = zip(*self.points)
                self.scatter.set_offsets(np.c_[xs, ys])
            else:
                self.scatter.set_offsets(np.c_[[], []])

        return self.scatter, self.robot_point, self.circle

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
                # Add point to list
                self.points.append((x, y))

                # Keep only last 50 points to avoid cluttering
                if len(self.points) > 50:
                    self.points.pop(0)

        except Exception as e:
            logger.error(f"Error updating plot: {e}")

    def close(self):
        """Clean up visualization resources."""
        try:
            self.should_stop = True
            if hasattr(self, "anim"):
                self.anim.event_source.stop()
            plt.close(self.fig)
            plt.ioff()  # Disable interactive mode
        except Exception as e:
            logger.error(f"Error closing plot: {e}")
