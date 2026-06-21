import numpy as np

class BallSimulation:
    def __init__(self, n_balls, dt=0.1, gravity=9.81, 
                 noise_std_x=1.0, noise_std_y=1.0, 
                 dropout_prob=0.1, restitution=0.8):
        """
        Simulates n balls thrown simultaneously.
        
        Args:
            n_balls (int): Number of balls to simulate.
            dt (float): Time step duration.
            gravity (float): Gravitational acceleration.
            noise_std_x (float): Standard deviation of observation noise in x.
            noise_std_y (float): Standard deviation of observation noise in y.
            dropout_prob (float): Probability of a complete sensor failure at a given time step.
            restitution (float): Coefficient of restitution for ground bounces.
        """
        self.n_balls = n_balls
        self.dt = dt
        self.g = gravity
        self.noise_std = np.array([noise_std_x, noise_std_y])
        self.dropout_prob = dropout_prob
        self.restitution = restitution
        
        # State: shape (n_balls, 4) -> x, y, vx, vy
        self.state = np.zeros((n_balls, 4))
        
        # Histories
        self.history = []
        self.observations = []
        
    def reset(self, initial_states):
        """
        Reset simulation with given initial states.
        
        Args:
            initial_states (np.ndarray): Array of shape (n_balls, 4).
        """
        self.state = np.array(initial_states, dtype=np.float64)
        self.history = [self.state.copy()]
        self.observations = [self.get_observation(self.state)]
        
    def step(self):
        """
        Advance the simulation by one time step dt.
        
        Returns:
            tuple: (current_state, current_observations)
        """
        # Update positions
        # x_new = x + vx * dt
        self.state[:, 0] += self.state[:, 2] * self.dt
        
        # y_new = y + vy * dt - 0.5 * g * dt^2
        self.state[:, 1] += self.state[:, 3] * self.dt - 0.5 * self.g * self.dt**2
        
        # Update velocities
        # vy_new = vy - g * dt
        self.state[:, 3] -= self.g * self.dt
        
        # Handle hitting the ground (y < 0)
        # Bounce the ball (multiple ups and downs)
        below_ground = self.state[:, 1] < 0
        if np.any(below_ground):
            # Bounce if moving downwards
            bounce_mask = below_ground & (self.state[:, 3] < 0)
            self.state[bounce_mask, 1] = -self.state[bounce_mask, 1]
            self.state[bounce_mask, 3] = -self.restitution * self.state[bounce_mask, 3]
            
            # Clip any remaining below-ground states to 0.0
            remaining_below = self.state[:, 1] < 0
            self.state[remaining_below, 1] = 0.0
            
        self.history.append(self.state.copy())
        
        obs = self.get_observation(self.state)
        self.observations.append(obs)
        
        return self.state.copy(), obs
        
    def get_observation(self, state):
        """
        Generates noisy, indistinguishable observations with possible dropouts.
        """
        # Sensor dropout: completely miss all balls
        if np.random.rand() < self.dropout_prob:
            return np.empty((0, 2))
        
        # Generate noisy positions
        obs = state[:, :2] + np.random.normal(0, self.noise_std, size=(self.n_balls, 2))
        
        # Shuffle to remove any implied ordering (indistinguishable observations)
        np.random.shuffle(obs)
        
        return obs
