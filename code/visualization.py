import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, x_lim=(-10, 60), y_lim=(0, 60)):
        """
        Setup the plotting environment.
        """
        plt.ion() # Interactive mode on
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Particle Filter: Multi-Ball Tracking')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        # Plot handles
        self.particles_scatter = self.ax.scatter([], [], s=2, color='gray', alpha=0.15, label='Particles')
        self.truth_scatter = self.ax.scatter([], [], s=60, color='blue', marker='o', edgecolors='black', label='True Positions')
        self.obs_scatter = self.ax.scatter([], [], s=120, color='red', marker='x', linewidths=2, label='Observations')
        self.est_scatter = self.ax.scatter([], [], s=90, color='green', marker='s', edgecolors='black', label='Estimated Positions')
        
        self.ax.legend(loc='upper left')
        
    def update(self, particles, truth, obs, est, title_suffix=""):
        """
        Updates the plot with the new frame data.
        """
        self.ax.set_title(f'Particle Filter: Multi-Ball Tracking {title_suffix}')
        
        # All particle positions
        p_flat = particles[..., :2].reshape(-1, 2)
        self.particles_scatter.set_offsets(p_flat)
        
        # True ball positions
        self.truth_scatter.set_offsets(truth[:, :2])
        
        # Observations (might be empty due to dropout)
        if obs is not None and len(obs) > 0:
            self.obs_scatter.set_offsets(obs)
            self.obs_scatter.set_visible(True)
        else:
            self.obs_scatter.set_visible(False)
            
        # Estimated cluster centers
        if est is not None and len(est) > 0:
            self.est_scatter.set_offsets(est)
            
        # Draw and pause
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)
        
    def keep_open(self):
        """
        Keeps the plot open after the simulation finishes.
        """
        plt.ioff()
        plt.show()
