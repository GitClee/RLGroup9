import numpy as np
from scipy.optimize import linear_sum_assignment

# ---- Particle filter: Condensation Algorithm ----

class ParticleFilter:

    def __init__(self, particle_amount, target_amount, filter_uncertainty, cov=1.0):
        self.particle_amount = particle_amount
        self.target_amount = target_amount
        self.filter_uncertainty = filter_uncertainty
        self.cov = cov

        self.particles = []
        self.weights = []

        # covariance matrix: shapes the normal distribution
        self.R = np.array([[self.cov ** 2, 0],
                      [0, self.cov ** 2]])

        self.inv_R = np.linalg.inv(self.R)

        self.mult = 1 / np.sqrt(np.linalg.det(2 * np.pi * self.R)**self.target_amount)


    # Initialise particle set with uniformly distributed particles
    def initialise_particles(self):

        # all weights initially set to 1/particle_amount
        self.weights = np.full((self.particle_amount, 1), 1 / self.particle_amount)

        self.particles = []

        for _ in range(self.particle_amount):

            particle = []

            # particle state length depends on target amount (for each ball location and velocity)
            for _ in range(self.target_amount):

                # assumption: x_0 and y_0 within [0,50]
                x = np.random.uniform(0, 50)
                y = np.random.uniform(0, 50)

                # assumption: v_x_0 and v_y_0 within [-25,25]
                v_x = np.random.uniform(-25, 25)
                v_y = np.random.uniform(-25, 25)

                particle.extend([x, y, v_x, v_y])

            self.particles.append(particle)

        self.particles = np.array(self.particles)

    # resample particles using weights
    def sample_particles(self):
        # get cumulative weights
        cumulative_weights = np.cumsum(self.weights[:, 0])
        cumulative_weights[-1] = 1.0

        sampled_particles = np.zeros((self.particle_amount, self.particles.shape[1]))

        for i in range(self.particle_amount):
            # random number in [0, 1[
            r = np.random.rand()

            for j in range(self.particle_amount):

                # sample first particle where random number is smaller than cumulative weight
                if cumulative_weights[j] >= r:
                    sampled_particles[i] = self.particles[j]
                    break

        self.particles = sampled_particles

    # apply transition model to propagate particles
    def propagate_particles(self, transition_model):
        propagated_particles = np.zeros_like(self.particles)

        for i, particle in enumerate(self.particles):
            p = particle.reshape(-1,1)
            propagated_particles[i] = transition_model.get_new_state(p, self.filter_uncertainty).flatten()

        self.particles = propagated_particles

    # evaluate particles using Gaussian distribution
    # multivariate normal distribution (see slides 200 page 17)
    def evaluate_particles(self, observation):

        new_weights = np.zeros(self.particle_amount)

        obs = observation.reshape(self.target_amount, 2)

        for p in range(self.particle_amount):

            particle = self.particles[p].reshape(self.target_amount, 4)
            particle_pos = particle[:, :2]

            cost = np.zeros((self.target_amount,self.target_amount))

            for i in range(self.target_amount):
                for j in range(self.target_amount):
                    diff = obs[i] - particle_pos[j]
                    cost[i, j] = diff.T @ self.inv_R @ diff

            # balls are not differentiable --> which position in particle belongs to which ball?
            # --> linear sum assignment problem
            rows, cols = linear_sum_assignment(cost)

            error = 0
            for i, j in zip(rows, cols):
                error += cost[i,j]

            new_weights[p] = self.mult * np.exp(-1/2 * error)

        sum_weights = np.sum(new_weights)

        # normalize so that sum(weights) = 1
        if sum_weights > 0:
            new_weights /= np.sum(new_weights)
        else:
            # if sum(weights) = 0 then set all weights to equal value
            new_weights = np.ones(self.particle_amount) /self.particle_amount

        self.weights = new_weights.reshape(-1, 1)

    def get_particles(self):
        return self.particles

    def get_weights(self):
        return self.weights