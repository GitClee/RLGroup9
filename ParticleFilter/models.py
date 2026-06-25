import numpy as np

# Transition model
class TransitionModel:

    def __init__(self, delta_t, target_amount):
        self.g = 9.81
        self.delta_t = delta_t
        self.target_amount = target_amount

        self.A = []
        self.B = []
        self.a = []

        self.calc_tran_model()


    def calc_tran_model(self):
        A_single = np.array([[1, 0, self.delta_t, 0],
                             [0, 1, 0, self.delta_t],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])

        B_single = np.array([[0, 0, 0],
                             [0, 0.5 * self.delta_t ** 2, 0],
                             [0, 0, 0],
                             [0, self.delta_t, 0]])

        a_single = np.array([[0, -self.g, 0]]).T

        self.A = np.zeros((4 * self.target_amount, 4 * self.target_amount))
        self.B = np.zeros((4 * self.target_amount, 3 * self.target_amount))
        self.a = np.zeros((3 * self.target_amount, 1))

        for i in range(self.target_amount):
            idx = i * 4
            col = i * 3
            self.A[idx:idx + 4, idx:idx + 4] = A_single

            self.B[idx:idx + 4, col:col + 3] = B_single

            self.a[col:col + 3] = a_single

    # calc new state using transition model
    def get_new_state(self, q, epsilon):
        eps = self.generate_epsilon(epsilon)
        next_state = self.A @ q + self.B @ self.a + eps

        # if ball hits ground stop movement
        for i in range(self.target_amount):
            y_pos = q[i * 4 + 1, 0]

            if y_pos <= 0:
                next_state[i * 4 + 1, 0] = 0
                next_state[i * 4 + 2, 0] = 0
                next_state[i * 4 + 3, 0] = 0

        return next_state

    # get error
    def generate_epsilon(self, epsilon):
        return np.random.normal(0, epsilon, (4 * self.target_amount, 1))

# Observation model
class ObservationModel:

    def __init__(self, target_amount):
        self.g = 9.81
        self.target_amount = target_amount

        self.C = []

        self.calc_obs_model()

    def calc_obs_model(self):
        # observation model
        self.C = np.zeros((2 * self.target_amount, 4 * self.target_amount))

        for i in range(self.target_amount):
            self.C[i * 2][i * 4] = 1
            self.C[i * 2 + 1][i * 4 + 1] = 1

    # calc observation
    def get_observation(self, q, delta):
        delt = self.generate_delta(delta)
        return self.C @ q + delt

    # get error
    def generate_delta(self, delta):
        return np.random.normal(0, delta, (2 * self.target_amount, 1))

