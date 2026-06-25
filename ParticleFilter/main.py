from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from simulation import Simulation
import numpy as np
from sklearn.cluster import KMeans

matplotlib.use("TkAgg")

# resets simulation and graph
def reset_simulation():
    global running, simulation, fig, ax
    global delta_t, particle_amount, n, drop_out_rate
    global epsilon_val, delta_val, epsilon_particle
    global xs, ys, meas_x, meas_y
    global balls, lines, vel_arrows

    if running:
        ani.pause()
        running = False

    # get new simulation with updated values
    simulation = Simulation(delta_t, particle_amount, n, drop_out_rate, epsilon_val, delta_val, epsilon_particle)
    simulation.generate_balls()

    q = simulation.get_q()

    # reset graph
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

    for i in range(n):
        line, = ax.plot([], [], 'b-')
        ball, = ax.plot([], [], 'go', markersize=15, zorder=3)
        arrow = ax.quiver(q[ i *4], q[ i * 4 +1], q[ i * 4 +2], q[ i * 4 +3], color='r', zorder=4)
        ball.set_data([q[ i *4]], [q[ i * 4 +1]])
        balls.append(ball)
        lines.append(line)
        vel_arrows.append(arrow)

    particle_plot.set_data([], [])
    estimation_plot.set_data([], [])
    meas_plot.set_data([], [])

    fig.canvas.draw()
    fig.canvas.flush_events()

# update simulation
def update(frame):
    global simulation, vel_arrows
    global xs, ys, meas_x, meas_y

    # if all balls landed stop updating
    landed = True

    q = simulation.get_q()

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

    # step simulation
    meas = simulation.step()

    # if measurement exists draw (dropout)
    if meas is not None:
        meas = meas.reshape(n, 2)
        for i in range(n):
            meas_x[i].append(meas[i, 0])
            meas_y[i].append(meas[i, 1])

        meas_plot.set_data(meas_x, meas_y)

    # get new state
    q = simulation.get_q()

    # draw new ball positions and update trajectory lines
    for i in range(n):
        x = q[i*4][0]
        y = q[i*4+1][0]

        xs[i].append(x)
        ys[i].append(y)

        lines[i].set_data(xs[i], ys[i])
        balls[i].set_data([x], [y])

    # get particles and plot
    particles, weights = simulation.get_particles()
    p = particles.reshape(particle_amount, n, 4)

    particle_plot.set_data(p[:,:,0].flatten(), p[:,:,1].flatten())

    # get particle cloud centers for ball position estimation
    pos = p[:, :, :2].reshape(-1, 2)

    cluster_weights = np.repeat(weights.flatten(), n)

    kmeans = KMeans(
        n_clusters=n,
        n_init=10,
        random_state=0).fit(pos, sample_weight=cluster_weights)

    estimation = kmeans.cluster_centers_

    estimation_plot.set_data(estimation[:, 0], estimation[:, 1])

    return lines + balls

# time steps for simulation
delta_t = 0.1
# Amount of balls in the simulation
n = 1
# drop out for sensor
drop_out_rate = 0.2
# add random noise to transition model (wind, etc.)
epsilon_val = 0.00
# add random noise to observation model (noise from sensor)
delta_val = 0.5
# uncertainty value for particle filter
epsilon_particle = 0.7
# number of particles
particle_amount= 1000

# for visualization
xs = None
ys = None
meas_x = None
meas_y = None
lines = []
balls = []
vel_arrows = []

simulation = None

# -- Setup Visualisation --
running = False
fig, ax = plt.subplots(figsize=(8, 6))

plt.subplots_adjust(bottom=0.4)
plt.grid()

ax.set_xlim(-50, 100)
ax.set_ylim(0, 100)

# Set up plots for particles, measurements and estimations
particle_plot, = ax.plot([], [], 'k.', alpha=0.3, markersize=2, zorder=4)
meas_plot, = ax.plot([],[], 'rx', markersize=6)
estimation_plot, = ax.plot([],[], 'mp', markersize=8, zorder=5)

# animate simulation
ani = FuncAnimation(fig, update, interval=10, blit=False)

reset_simulation()
fig.canvas.draw_idle()

# Setup sliders and buttons for adjusting variables

ax_balls = plt.axes((0.2, 0.3, 0.5, 0.03))
ax_time_step = plt.axes((0.2, 0.25, 0.5, 0.03))
ax_dropout = plt.axes((0.2, 0.2, 0.5, 0.03))
ax_epsilon_particles = plt.axes((0.2, 0.15, 0.5, 0.03))
ax_particle_amount = plt.axes((0.2, 0.1, 0.5, 0.03))
ax_noise_transition = plt.axes((0.2, 0.05, 0.5, 0.03))
ax_noise_observation = plt.axes((0.2, 0, 0.5, 0.03))
ax_start = plt.axes((0.8, 0.3, 0.15, 0.04))
ax_reset = plt.axes((0.8, 0.25, 0.15,0.04))

ball_slider = Slider(ax_balls,"Amount of balls",1,10,
                     valinit=n, valstep=1)
time_step_slider = Slider(ax_time_step,"Time step", 0.01,1,
                          valinit=delta_t, valstep=0.01)
drop_out_slider = Slider(ax_epsilon_particles,"Sensor dropout",0,0.99,
                         valinit=drop_out_rate,valstep=0.01)
epsilon_particle_slider = Slider(ax_dropout, "Filter uncertainty",0,5,
                                 valinit=epsilon_particle,valstep=0.01)
particle_amount_slider = Slider(ax_particle_amount,"Amount of particles",1,2000,
                                valinit=particle_amount, valstep=1.0)
noise_transition_slider = Slider(ax_noise_transition,"Environmental noise",0,1,
                                valinit=epsilon_val, valstep=0.01)
noise_observation_slider = Slider(ax_noise_observation,"Sensor noise",0,3,
                                valinit=delta_val, valstep=0.01)

reset_button = Button(ax_reset,"Reset")
start_button = Button(ax_start,"Start")

# functions called by sliders
def update_balls(value):
    global n
    n = int(value)

def update_time_step(value):
    global delta_t
    delta_t = float(value)

def update_dropout(value):
    global drop_out_rate
    drop_out_rate = float(value)

def update_epsilon_particles(value):
    global epsilon_particle
    epsilon_particle = float(value)

def update_particle_amount(value):
    global particle_amount
    particle_amount = int(value)

def update_noise_transition(value):
    global epsilon_val
    epsilon_val = float(value)

def update_noise_observation(value):
    global delta_val
    delta_val = float(value)

ball_slider.on_changed(update_balls)
time_step_slider.on_changed(update_time_step)
drop_out_slider.on_changed(update_dropout)
epsilon_particle_slider.on_changed(update_epsilon_particles)
particle_amount_slider.on_changed(update_particle_amount)
noise_transition_slider.on_changed(update_noise_transition)
noise_observation_slider.on_changed(update_noise_observation)

# functions called by buttons
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

    start_button.label.set_text("Start")

    fig.canvas.draw_idle()

    reset_simulation()

start_button.on_clicked(start)
reset_button.on_clicked(reset)

# start/stop function for spacebar
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