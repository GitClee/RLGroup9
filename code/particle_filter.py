import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

class JointParticleFilter:
    def __init__(self, n_particles, n_balls, initial_bounds, 
                 process_noise_std, obs_noise_std):
        """
        Joint State Particle Filter for multi-target tracking.
        
        Args:
            n_particles (int): Number of particles.
            n_balls (int): Number of indistinguishable balls.
            initial_bounds (tuple): ((min_x, max_x), (min_y, max_y), (min_vx, max_vx), (min_vy, max_vy))
            process_noise_std (list): [std_x, std_y, std_vx, std_vy]
            obs_noise_std (list): [std_x, std_y]
        """
        self.N = n_particles
        self.n_balls = n_balls
        self.obs_noise_std = np.array(obs_noise_std)
        self.process_noise_std = np.array(process_noise_std)
        
        # State tensor: shape (N, n_balls, 4) representing (x, y, vx, vy)
        self.particles = np.zeros((self.N, self.n_balls, 4))
        self.weights = np.ones(self.N) / self.N
        
        # Initialize uniformly within the specified bounds
        (x_b, y_b, vx_b, vy_b) = initial_bounds
        self.particles[..., 0] = np.random.uniform(x_b[0], x_b[1], (self.N, self.n_balls))
        self.particles[..., 1] = np.random.uniform(y_b[0], y_b[1], (self.N, self.n_balls))
        self.particles[..., 2] = np.random.uniform(vx_b[0], vx_b[1], (self.N, self.n_balls))
        self.particles[..., 3] = np.random.uniform(vy_b[0], vy_b[1], (self.N, self.n_balls))
        
    def predict(self, dt, gravity=9.81, restitution=0.8):
        """
        Transition model applying physical equations of motion and process noise.
        """
        # Sample process noise for all particles and balls
        noise = np.random.normal(0, self.process_noise_std, size=(self.N, self.n_balls, 4))
        
        # Update positions
        self.particles[..., 0] += self.particles[..., 2] * dt + noise[..., 0]
        self.particles[..., 1] += self.particles[..., 3] * dt - 0.5 * gravity * dt**2 + noise[..., 1]
        
        # Update velocities
        self.particles[..., 2] += noise[..., 2]
        self.particles[..., 3] += -gravity * dt + noise[..., 3]
        
        # Ground collision logic (bounce)
        below_ground = self.particles[..., 1] < 0
        bounce_mask = below_ground & (self.particles[..., 3] < 0)
        
        self.particles[bounce_mask, 1] = -self.particles[bounce_mask, 1]
        self.particles[bounce_mask, 3] = -restitution * self.particles[bounce_mask, 3]
        
        # Clip any remaining below-ground states to 0.0
        remaining_below = self.particles[..., 1] < 0
        self.particles[remaining_below, 1] = 0.0
        
    def update(self, observations):
        """
        Update particle weights based on optimal assignment with indistinguishable observations.
        """
        m = observations.shape[0]
        
        # Handle sensor dropout scenario
        if m == 0:
            # No observations to update weights, retain previous uncertainty distribution
            return
            
        var = np.mean(self.obs_noise_std**2)
        
        for i in range(self.N):
            pred_pos = self.particles[i, :, :2]
            
            # Compute distance matrix between all predictions and observations
            # Shape: (n_balls, m)
            diff = pred_pos[:, np.newaxis, :] - observations[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Use Hungarian Algorithm to find the optimal association
            # minimizing the sum of squared errors
            row_ind, col_ind = linear_sum_assignment(dist_sq)
            cost = dist_sq[row_ind, col_ind].sum()
            
            # Multiply weight by likelihood of this optimal assignment
            self.weights[i] *= np.exp(-cost / (2 * var))
            
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum < 1e-15:
            # If all weights collapse (e.g. extremely unlikely observation),
            # re-initialize uniformly to prevent NaNs
            self.weights = np.ones(self.N) / self.N
        else:
            self.weights /= weight_sum
            
    def resample(self):
        """
        Systematic Resampling step to discard low-weight particles.
        """
        N_eff = 1.0 / np.sum(self.weights**2)
        
        # Resample only if Effective Sample Size is too low
        if N_eff < self.N / 2.0:
            positions = (np.arange(self.N) + np.random.uniform(0, 1)) / self.N
            indexes = np.zeros(self.N, dtype=int)
            cumulative_sum = np.cumsum(self.weights)
            
            i, j = 0, 0
            while i < self.N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
                    
            self.particles = self.particles[indexes]
            self.weights = np.ones(self.N) / self.N

    def estimate(self):
        """
        Estimates expected positions of the $n$ balls.
        Resolves "label switching" and multimodality using K-Means clustering 
        over the entire particle cloud positional predictions.
        
        Returns:
            np.ndarray: Array of shape (n_balls, 2) containing estimated (x,y) positions.
        """
        # Reshape to treat all ball predictions as individual 2D points
        all_positions = self.particles[..., :2].reshape(-1, 2)
        weights_expanded = np.repeat(self.weights, self.n_balls)
        
        # Cluster the points to find the highest density regions (the multi-modal peaks)
        kmeans = KMeans(n_clusters=self.n_balls, n_init=10, random_state=None)
        kmeans.fit(all_positions, sample_weight=weights_expanded)
        
        return kmeans.cluster_centers_
