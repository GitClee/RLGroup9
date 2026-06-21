import numpy as np
import time
from simulation import BallSimulation
from particle_filter import JointParticleFilter
from visualization import Visualizer

def main():
    # --- Configuration ---
    n_balls = 3
    dt = 0.2
    steps = 100
    
    # Simulation noise and dropout parameters
    sim_noise_std = 2.0
    dropout_prob = 0.15 # 15% chance of dropping out all observations for a step
    
    # Particle Filter parameters
    n_particles = 1000
    pf_process_noise_std = [0.2, 0.2, 0.5, 0.5] # Adds robustness to unmodeled dynamics
    pf_obs_noise_std = [sim_noise_std, sim_noise_std]
    
    # Filter's prior knowledge about initial state (50x50m area)
    init_bounds = (
        (0.0, 50.0),   # x
        (0.0, 50.0),   # y
        (-5.0, 20.0),  # vx
        (-5.0, 20.0)   # vy
    )
    
    # --- Initialization ---
    print(f"Initializing simulation with {n_balls} balls...")
    sim = BallSimulation(n_balls=n_balls, dt=dt, 
                         noise_std_x=sim_noise_std, 
                         noise_std_y=sim_noise_std, 
                         dropout_prob=dropout_prob)
    
    # Random true initial states within a subset of the bounds
    true_init = np.zeros((n_balls, 4))
    true_init[:, 0] = np.random.uniform(5, 25, n_balls) # Start on the left side
    true_init[:, 1] = np.random.uniform(10, 40, n_balls) 
    true_init[:, 2] = np.random.uniform(5, 15, n_balls) # Move right
    true_init[:, 3] = np.random.uniform(0, 15, n_balls) # Move up
    
    sim.reset(true_init)
    
    print(f"Initializing Joint Particle Filter with {n_particles} particles...")
    pf = JointParticleFilter(n_particles=n_particles, 
                             n_balls=n_balls, 
                             initial_bounds=init_bounds,
                             process_noise_std=pf_process_noise_std,
                             obs_noise_std=pf_obs_noise_std)
                             
    # x and y limits slightly expanded to let balls fly far
    viz = Visualizer(x_lim=(-10, 150), y_lim=(-10, 80))
    
    print("Starting simulation loop...")
    for t in range(steps):
        # 1. Advance true simulation and generate observations
        true_state, obs = sim.step()
        
        # 2. Predict particle states forward in time
        pf.predict(dt, restitution=sim.restitution)
        
        # 3. Update particle weights based on the optimal assignment likelihood
        pf.update(obs)
        
        # 4. Resample particles if weights degenerate
        pf.resample()
        
        # 5. Estimate target positions using K-Means clustering over all particles
        est_positions = pf.estimate()
        
        # 6. Update visualizer
        status = f" | Step {t+1}/{steps}"
        if len(obs) == 0:
            status += " [SENSOR DROPOUT]"
        
        viz.update(pf.particles, true_state, obs, est_positions, title_suffix=status)

    print("Simulation complete. Close the plot window to exit.")
    viz.keep_open()

if __name__ == "__main__":
    main()
