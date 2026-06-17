import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

running = False

# gravity acceleration
g = 9.81
# time steps for simulation
delta_t = 0.01
# drop out for sensor
drop_out_rate = 0.8
# add random noise to transition model (wind, etc.)
epsilon_val = 0.00
# add random noise to observation model (noise from sensor)
delta_val = 0.5
# uncertainty value for particle filter
epsilon_particle = 0.5
# number of particles
particle_amount= 1000

# observation model
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])


# Transition model
def calc_tran_model(delta_t, g):
    A = np.array([[1, 0, delta_t, 0],
                   [0, 1, 0, delta_t],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    B = np.array([[0, 0, 0],
                   [0, 0.5 * delta_t ** 2, 0],
                   [0, 0, 0],
                   [0, delta_t, 0]])

    a = np.array([[0, -g, 0]]).T

    return A, B, a

# add error
def generate_epsilon(epsilon_val):
     return np.random.normal(0, epsilon_val, (4,1))

def generate_delta(delta_val):
    return np.random.normal(0, delta_val, (2, 1))

# initial positions and velocity
x_0 = 1
y_0 = 1
v_x_0 = 20
v_y_0 = 20

A, B, a = calc_tran_model(delta_t, g)

q = np.array([[x_0, y_0, v_x_0, v_y_0]]).T

# calc new state using transition model
def get_new_state(q, A, B, a, epsilon_val=epsilon_val):
    epsilon = generate_epsilon(epsilon_val)
    return A @ q + B @ a + epsilon

# calc observation
def get_observation(q, C):
    delta = generate_delta(delta_val)
    return C @ q + delta

xs = [x_0]
ys = [y_0]
meas_x = []
meas_y = []

fig, ax = plt.subplots()

ax.set_xlim(0, 90)
ax.set_ylim(0, 90)

line, = ax.plot([],[], 'b-')
particle_plot, = ax.plot([], [], 'k.', alpha=0.3, markersize=2)
ball, = ax.plot([],[], 'go', markersize=10)
meas_plot, = ax.plot([],[], 'rx', markersize=6)

ball.set_data([x_0], [y_0])

# ---- Particle filter: Condensation Algorithm ----

# Create particle set with n uniformly distributed particles
def get_particles(n):
    # weights initially the same
    weights = np.full((n, 1), 1/n)

    # assumption: x_0 and y_0 within [0,50]
    x_particles = np.random.uniform(0, 50, (n,1))
    y_particles = np.random.uniform(0, 50, (n,1))

    # assumption: v_x_0 and v_y_0 within [-25,25]
    v_x_particles = np.random.uniform(-25, 25, (n,1))
    v_y_particles = np.random.uniform(-25, 25, (n,1))

    particles = np.hstack([x_particles, y_particles, v_x_particles, v_y_particles])

    return particles, weights

def sample_particles(particles, weights, n):
    cumulative_weights = np.cumsum(weights[:, 0])
    cumulative_weights[-1] = 1.0

    sampled_particles = np.zeros((n,4))

    for i in range(n):
        r = np.random.rand()

        for j in range(particles.shape[0]):
            if cumulative_weights[j] >= r:
                sampled_particles[i] = particles[j]
                break

    return sampled_particles, (np.ones(n) / n).reshape(n, 1)

def propagate_particles(particles):
    propagated_particles = np.zeros_like(particles)

    for i, particle in enumerate(particles):
        part = particle.reshape(4,1)
        propagated_particles[i] = get_new_state(part, A, B, a, epsilon_particle).flatten()

    return propagated_particles

# evaluate particles using Gaussian distribution
# mulitvariate normal distribution slides 200 page 17
def evaluate_particles(particles, observation):
    particle_pos = particles[:, :2]

    # noise covariance
    R = np.array([[1.0**2, 0],
                  [0, 1.0**2]])

    diffs = observation.flatten() - particle_pos
    inv_R = np.linalg.inv(R)

    exponent = -1/2 * np.sum((diffs @ inv_R) * diffs, axis=1)

    mult = 1 /np.sqrt(np.linalg.det(2 * np.pi * R))

    new_weights = mult * np.exp(exponent)

    new_weights /= np.sum(new_weights)

    return new_weights.reshape(particle_amount, 1)


particles, weights = get_particles(particle_amount)


def update(frame):
    global q, weights, particles
    if q[1][0] <= 0 or not running:
      ani.pause()
      return line, ball

    # simulation
    q = get_new_state(q, A, B, a)

    # sample and propagate particles
    particles, weights = sample_particles(particles, weights, particle_amount)
    particles = propagate_particles(particles)

    # measurement
    if np.random.rand() > drop_out_rate:
        meas = get_observation(q, C)
        meas_x.append(meas[0])
        meas_y.append(meas[1])

        # if meas available evaluate
        weights = evaluate_particles(particles, meas)

    xs.append(q[0][0])
    ys.append(q[1][0])

    line.set_data(xs, ys)
    ball.set_data(q[0], q[1])
    meas_plot.set_data(meas_x, meas_y)
    particle_plot.set_data(particles[:,0], particles[:,1])

    return line, ball, meas_plot

ani = FuncAnimation(fig, update, interval=10, blit=False)

fig.canvas.draw_idle()


def on_key(event):
     global running
     if event.key == " ":
          ani.resume()
          running = True

fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()





