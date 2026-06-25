import numpy as np
from models import TransitionModel, ObservationModel
from ParticleFilter import ParticleFilter

class Simulation:

    def __init__(self,
                 delta_t,
                 particle_amount,
                 target_amount,
                 drop_out_rate,
                 epsilon,
                 delta,
                 filter_uncertainty):

        self.target_amount = target_amount
        self.drop_out_rate = drop_out_rate
        self.epsilon = epsilon
        self.delta = delta
        self.q = []

        self.trans_model = TransitionModel(delta_t, target_amount)
        self.obs_model = ObservationModel(target_amount)
        self.particle_filter = ParticleFilter(particle_amount, target_amount, filter_uncertainty)
        self.particle_filter.initialise_particles()

    # generate random ball positions and velocities
    def generate_balls(self):
        state = []
        for i in range(self.target_amount):
            x = np.random.uniform(0, 50)
            y = np.random.uniform(0, 50)
            v_x = np.random.uniform(-25, 25)
            v_y = np.random.uniform(-25, 25)
            state.extend([x, y, v_x, v_y])
        self.q = np.array(state).reshape(-1, 1)

    def step(self):
        self.q = self.trans_model.get_new_state(self.q, self.epsilon)

        # sample and propagate particles
        self.particle_filter.sample_particles()
        self.particle_filter.propagate_particles(self.trans_model)

        # measurement
        if np.random.rand() > self.drop_out_rate:

            meas = self.obs_model.get_observation(self.q, self.delta)

            # if meas available evaluate
            self.particle_filter.evaluate_particles(meas)

            return meas

        return None

    def get_q(self):
        return self.q

    def get_particles(self):
        return self.particle_filter.get_particles(), self.particle_filter.get_weights()

