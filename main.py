import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

running = False

# gravity acceleration
g = 9.81
# time steps for simulation
delta_t = 0.01
delta_t_pending = delta_t
# Amount of balls in the simulation
n = 1
n_pending = n
# drop out for sensor
drop_out_rate = 0.5
drop_out_pending = drop_out_rate
# add random noise to transition model (wind, etc.)
epsilon_val = 0.00
epsilon_val_pending = epsilon_val
# add random noise to observation model (noise from sensor)
delta_val = 0.5
# uncertainty value for particle filter
epsilon_particle = 0.5
epsilon_particle_pending = epsilon_particle
# number of particles
particle_amount= 1000
particle_amount_pending = particle_amount

q = None
A = None
B = None
a = None
C = None
xs = None
ys = None
meas_x = None
meas_y = None
particles = None
weights = None
lines = []
balls = []
vel_arrows = []

# Transition model
def calc_tran_model(delta_t, g, n):
    A_single = np.array([[1, 0, delta_t, 0],
                   [0, 1, 0, delta_t],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    B_single = np.array([[0, 0, 0],
                   [0, 0.5 * delta_t ** 2, 0],
                   [0, 0, 0],
                   [0, delta_t, 0]])


    a_single = np.array([[0, -g, 0]]).T

    A = np.zeros((4*n, 4*n))
    B = np.zeros((4*n, 3*n))
    a = np.zeros((3*n, 1))

    for i in range(n):

        idx = i*4
        col = i*3
        A[idx:idx+4, idx:idx+4] = A_single

        B[idx:idx+4, col:col+3] = B_single

        a[col:col+3] = a_single

    return A, B, a

def calc_obs_model(n):
    # observation model
    C = np.zeros((2*n, 4*n))

    for i in range(n):
        C[i*2][i*4] = 1
        C[i*2+1][i*4 +1] = 1

    return C

# add error
def generate_epsilon(epsilon_val, n):
     return np.random.normal(0, epsilon_val, (4*n,1))

def generate_delta(delta_val, n):
    return np.random.normal(0, delta_val, (2*n, 1))

def generate_balls(n):
    q = []
    for i in range(n):
        x = np.random.uniform(0,50)
        y = np.random.uniform(0,50)
        v_x = np.random.uniform(-25, 25)
        v_y = np.random.uniform(-25, 25)
        q.extend([x, y, v_x, v_y])
    return np.array(q).reshape(-1,1)

# calc new state using transition model
def get_new_state(q, A, B, a, n, epsilon_val=epsilon_val):
    epsilon = generate_epsilon(epsilon_val,n)
    next_state = A @ q + B @ a + epsilon

    # if ball hits ground stop movement
    for i in range(n):
        y_pos = q[i*4+1, 0]

        if y_pos <= 0:
            next_state[i*4+1, 0] = 0
            next_state[i*4+2, 0] = 0
            next_state[i*4+3, 0] = 0

    return next_state

# calc observation
def get_observation(q, C, n, delta_val=delta_val):
    delta = generate_delta(delta_val, n)
    return C @ q + delta

# ---- Particle filter: Condensation Algorithm ----

# Initialise particle set with n uniformly distributed particles
def get_particles(particle_amount, n):
    # weights initially the same
    weights = np.full((particle_amount, 1), 1 / particle_amount)

    particles = []

    for _ in range(particle_amount):

        particle = []

        for _ in range(n):
            # assumption: x_0 and y_0 within [0,50]
            x = np.random.uniform(0, 50)
            y = np.random.uniform(0, 50)

            # assumption: v_x_0 and v_y_0 within [-25,25]
            v_x = np.random.uniform(-25, 25)
            v_y = np.random.uniform(-25, 25)

            particle.extend([x, y, v_x, v_y])

        particles.append(particle)

    particles = np.array(particles)

    return particles, weights

def sample_particles(particles, weights, n):
    cumulative_weights = np.cumsum(weights[:, 0])
    cumulative_weights[-1] = 1.0

    sampled_particles = np.zeros((n,particles.shape[1]))

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
        part = particle.reshape(-1,1)
        propagated_particles[i] = get_new_state(part, A, B, a, n, epsilon_particle).flatten()

    return propagated_particles

# evaluate particles using Gaussian distribution
# mulitvariate normal distribution slides 200 page 17
def evaluate_particles(particles, observation, n):

    new_weights = np.zeros(particles.shape[0])

    # noise covariance
    R = np.array([[1.0**2, 0],
                  [0, 1.0**2]])

    inv_R = np.linalg.inv(R)

    mult = 1 /np.sqrt(np.linalg.det(2 * np.pi * R)**n)

    obs = observation.reshape(n, 2)

    for p in range(particles.shape[0]):

        particle = particles[p].reshape(n, 4)
        particle_pos = particle[:, :2]

        cost = np.zeros((n,n))

        for i in range(n):
            for j in range(n):

                diff = obs[i] - particle_pos[j]
                cost[i,j] = diff.T @ inv_R @ diff

        # check which predictions match what observation
        # (balls are not differentiable)
        rows, cols = linear_sum_assignment(cost)

        error = 0
        for i, j in zip(rows, cols):
            error += cost[i,j]

        new_weights[p] = mult * np.exp(-1/2 * error)

    new_weights /= np.sum(new_weights)

    return new_weights.reshape(-1, 1)

def reset_simulation():
    global q, A, B, a, C, xs, ys, meas_x, meas_y, particles, weights, lines, balls
    global balls, lines, vel_arrows, ani, running

    if 'ani' in globals():
        ani.pause()
        running = False


    # initial positions and velocity
    q = generate_balls(n)

    A, B, a = calc_tran_model(delta_t, g, n)
    C = calc_obs_model(n)

    xs = [[] for _ in range(n)]
    ys = [[] for _ in range(n)]
    meas_x = [[] for _ in range(n)]
    meas_y = [[] for _ in range(n)]

    for l in lines:
        l.remove()
    for b in balls:
        b.remove()
    for arrow in vel_arrows:
        arrow.remove()

    balls = []
    lines = []
    vel_arrows = []

    particles, weights = get_particles(particle_amount, n)

    for i in range(n):
        line, = ax.plot([], [], 'b-')
        ball, = ax.plot([], [], 'go', markersize=10, zorder=3)
        arrow = ax.quiver(q[i*4], q[i*4+1], q[i*4+2], q[i*4+3], color='r', zorder=4)
        ball.set_data([q[i*4]], [q[i*4+1]])
        balls.append(ball)
        lines.append(line)
        vel_arrows.append(arrow)

    particle_plot.set_data([], [])
    meas_plot.set_data([], [])

    fig.canvas.draw()
    fig.canvas.flush_events()

fig, ax = plt.subplots()

plt.subplots_adjust(bottom=0.3)
plt.grid

ax.set_xlim(0, 90)
ax.set_ylim(0, 90)

ax_balls = plt.axes((0.2, 0.15, 0.5, 0.03))
ax_time_step = plt.axes((0.2, 0.1, 0.5, 0.03))
ax_dropout = plt.axes((0.2, 0, 0.5, 0.03))
ax_epsilon_particles = plt.axes((0.2, 0.05, 0.5, 0.03))
ax_start = plt.axes((0.8, 0.15, 0.15, 0.04))
ax_reset = plt.axes((0.8, 0.1, 0.15,0.04))

ball_slider = Slider(
    ax_balls,
    "Amount of balls",
    1,
    10,
    valinit=1,
    valstep=1
)

time_step_slider = Slider(
    ax_time_step,
    "Time step",
    valmin= 0.01,
    valmax= 1,
    valinit=0.01,
    valstep=0.01
)

drop_out_slider = Slider(
    ax_epsilon_particles,
    "Sensor dropout",
    valmin=0,
    valmax=0.99,
    valinit=0.5,
    valstep=0.01
)

epsilon_particle_slider = Slider(
    ax_dropout,
    "Filter uncertainty",
    valmin=0,
    valmax=1,
    valinit=0.5,
    valstep=0.01
)

reset_button = Button(
    ax_reset,
    "Reset"
)

start_button = Button(
    ax_start,
    "Start"
)

def update_balls(value):
    global n_pending
    n_pending = int(value)

def update_time_step(value):
    global delta_t_pending
    delta_t_pending = float(value)

def update_dropout(value):
    global drop_out_pending
    drop_out_pending = float(value)

def update_epsilon_particles(value):
    global epsilon_particle_pending
    epsilon_particle_pending = float(value)

def start(event):
    global running
    if running:
        ani.pause()
        start_button.label.set_text("Start")
        fig.canvas.draw_idle()
    else:
        ani.resume()
        start_button.label.set_text("Pause")
        fig.canvas.draw_idle()
    running = not running

def reset(event):
    global n, n_pending, delta_t, delta_t_pending, drop_out_rate, drop_out_pending
    global epsilon_particle_pending, epsilon_particle

    n = n_pending
    delta_t = delta_t_pending
    drop_out_rate = drop_out_pending
    epsilon_particle = epsilon_particle_pending
    start_button.label.set_text("Start")
    fig.canvas.draw_idle()
    reset_simulation()


ball_slider.on_changed(update_balls)
time_step_slider.on_changed(update_time_step)
drop_out_slider.on_changed(update_dropout)
epsilon_particle_slider.on_changed(update_epsilon_particles)
start_button.on_clicked(start)
reset_button.on_clicked(reset)

particle_plot, = ax.plot([], [], 'k.', alpha=0.3, markersize=2, zorder=1)
meas_plot, = ax.plot([],[], 'rx', markersize=6)

reset_simulation()


def update(frame):
    global q, weights, particles
    global vel_arrows

    landed = True

    for i in range(n):
        if q[i*4 +1][0] > 0:
            landed = False
            break

    if landed or not running:
      ani.pause()
      return lines + balls

    # remove arrows
    for arrow in vel_arrows:
        arrow.remove()
    vel_arrows = []

    # simulation
    q = get_new_state(q, A, B, a, n)

    # sample and propagate particles
    particles, weights = sample_particles(particles, weights, particle_amount)
    particles = propagate_particles(particles)

    # measurement
    if np.random.rand() > drop_out_rate:

        meas = get_observation(q, C, n)
        meas = meas.reshape(n,2)
        for i in range(n):
            meas_x[i].append(meas[i, 0])
            meas_y[i].append(meas[i, 1])

        # if meas available evaluate
        weights = evaluate_particles(particles, meas, n)


    for i in range(n):
        x = q[i*4][0]
        y = q[i*4+1][0]

        xs[i].append(x)
        ys[i].append(y)

        lines[i].set_data(xs[i], ys[i])
        balls[i].set_data([x], [y])

    meas_plot.set_data(meas_x, meas_y)

    p = particles.reshape(particle_amount, n, 4)

    particle_plot.set_data(p[:,:,0].flatten(), p[:,:,1].flatten())

    return lines + balls

ani = FuncAnimation(fig, update, interval=10, blit=False)

fig.canvas.draw_idle()


def on_key(event):
     global running
     if event.key == " ":
        if running:
            ani.pause()
            start_button.label.set_text("Start")
            fig.canvas.draw_idle()
        else:
            ani.resume()
            start_button.label.set_text("Pause")
            fig.canvas.draw_idle()
        running = not running

fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()





