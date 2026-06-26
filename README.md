# Sensor Fusion Particle Filter: Ball Tracking

## 1. General Overview of the Program
This program simulates and estimates the trajectories of $n$ balls thrown simultaneously in a 2D plane (x, y) with gravity. It implements a Particle Filter to track the balls' positions and velocity vectors based solely on noisy, intermittent sensor observations of their positions. 

Key features include:
- A dynamic physics simulation where balls follow parabolic trajectories under gravity.
- An imperfect sensor model that only measures position (not velocity), includes Gaussian noise, and frequently drops out.
- A Particle Filter that estimates the state (position and velocity) of all $n$ balls despite the balls being indistinguishable from the sensor's perspective.
- An interactive visualization using `matplotlib`, allowing real-time parameter changes and visual tracking of the true states, noisy measurements, and the estimated particle cloud.

## 2. How to run
To start this program, run the *main.py* file.
In the simulation window, press the *Start* button to start the simulation. 
Variables can be adjusted using the sliders at the bottom of the window. To apply the value changes, press the *Reset* button.
Ball positions and velocities are randomized with each reset.
To add specific ball positions and velocities, enter the values in the corresponding text boxes and press the *Add* Button.

## 3. Explanation of the Particle Filter
A Particle Filter is a sequential Monte Carlo method used to estimate the internal state of a dynamic system from noisy partial observations. In this problem, it is used because the relationship between the measurements and the states is subject to uncertainty, and the indistinguishability of the balls creates a highly non-linear, multimodal probability distribution.

The algorithm (Condensation Algorithm) follows these steps:
1. **Initialization**: We generate a set of `M` particles. Each particle represents a "guess" of the entire system state (the positions and velocities of all $n$ balls).
2. **Resampling**: We randomly draw a new generation of `M` particles from the current set, where particles with higher weights are more likely to be chosen. This focuses our computational effort on the most probable states.
3. **Propagation (Prediction)**: We move each particle forward in time using our physical transition model (gravity, velocity). We also inject process noise (`epsilon_particle`) to ensure the particles explore the state space and don't collapse into a single point.
4. **Evaluation (Update)**: When a sensor measurement is available, we evaluate how likely each particle's guess is. Since balls are indistinguishable, we use the **Hungarian Algorithm** to find the optimal assignment between a particle's predicted ball positions and the observed measurements, minimizing the distance. Particles whose predictions closely match the observations are assigned higher weights.


## 4. Used Libraries
- **`numpy`**: Used for matrix operations, vector math, and random number generation (Gaussian noise, uniform initialization). It is crucial for handling the state vectors, transition matrices, and multivariate Gaussian probability calculations efficiently.
- **`scipy.optimize` (`linear_sum_assignment`)**: Used to solve the Assignment Problem (Hungarian Algorithm). It finds the optimal one-to-one matching between the indistinguishable predicted balls and the sensor observations to minimize the total assignment cost (distance).
- **`matplotlib.pyplot` & `matplotlib.animation` (`FuncAnimation`)**: Used to render the 2D visualization and animate the simulation in real-time.
- **`matplotlib.widgets` (`Slider`, `Button`, `RadioButtons`)**: Used to create the interactive UI elements (sliders for balls and time steps, buttons for start/pause/reset).

## 5. Formulas and Matrices

### State Vector
For $n$ balls, the state vector $q$ is a $4n \times 1$ column vector. For a single ball $i$, the state is:

$$
q_i = \begin{bmatrix} 
  x \\\\
  y \\\\
  v_x \\\\
  v_y 
\end{bmatrix}
$$

### Transition Model
The transition model predicts the next state:

$$
q_{t} = A \cdot q_{t-1} + B \cdot a + \epsilon
$$

Where:

**$A$ (State Transition Matrix)**: Updates position based on velocity.

$$
A_{single} = \begin{bmatrix} 
  1 & 0 & \Delta t & 0 \\\\
  0 & 1 & 0 & \Delta t \\\\
  0 & 0 & 1 & 0 \\\\
  0 & 0 & 0 & 1 
\end{bmatrix}
$$

**$B$ (Control Input Matrix) & $a$ (Acceleration/Gravity)**: Updates position and velocity based on gravity ($g = 9.81$).

$$
B_{single} = \begin{bmatrix} 
  0 & 0 & 0 \\\\
  0 & 0.5 \Delta t^2 & 0 \\\\
  0 & 0 & 0 \\\\
  0 & \Delta t & 0 
\end{bmatrix}, \quad 
a_{single} = \begin{bmatrix} 
  0 \\\\
  -g \\\\
  0 
\end{bmatrix}
$$

**$\epsilon$**: Process noise added to the system.

> [!NOTE]
> For $n$ balls, these matrices are expanded into block-diagonal matrices of sizes $4n \times 4n$, $4n \times 3n$, and $3n \times 1$ respectively.

### Observation Model
The observation model projects the state into the measurement space (we only observe positions $x, y$, not velocities $v_x, v_y$):

$$
z_t = C \cdot q_t + \delta
$$

Where:

**$C$ (Observation Matrix)**: Extracts the $x$ and $y$ coordinates.

$$
C_{single} = \begin{bmatrix} 
  1 & 0 & 0 & 0 \\\\
  0 & 1 & 0 & 0 
\end{bmatrix}
$$

**$\delta$**: Sensor noise (Gaussian).

### Particle Evaluation (Weighting)
The weight of a particle is updated based on a Multivariate Gaussian distribution comparing the prediction and observation:

$$
\text{weight} \propto \frac{1}{\sqrt{|2\pi R|^n}} \exp\left( -\frac{1}{2} \text{error} \right)
$$

Where $R$ is the sensor noise covariance matrix, and the $\text{error}$ is the minimized Mahalanobis distance sum calculated via the Hungarian algorithm.

## 6. Interactive Variables and Parameters

### Variables Controlled via UI Sliders
These variables can be changed dynamically while the program is running using the UI sliders:
- **Amount of balls**: 
  - *What it does*: Changes the number of balls being tracked simultaneously.
  - *Behavior*: Increasing this heavily impacts performance because the state matrices grow ($4n \times 4n$) and the Hungarian algorithm has to solve an $n \times n$ matching problem for *every* particle in every frame. Visually, you will see more balls and a denser particle cloud.
- **Time step**: 
  - *What it does*: Controls the discrete time interval between simulation frames.
  - *Behavior*: Increasing the time step makes the simulation run faster but causes the physics approximation to become less accurate. If it's too high, balls might clip through the ground or move erratically because the discrete steps are too large to smoothly approximate the continuous parabola.
- **Sensor dropout**: 
  - *What it does*: The probability that the sensor fails to provide a reading in a given frame.
  - *Behavior*: If increased, the particle cloud will rely entirely on its internal physics predictions for longer periods. This causes the cloud to spread out and expand as uncertainty grows, until a new measurement pulls it back together.
- **Filter uncertainty**: 
  - *What it does*: The noise injected into the particles during propagation (`epsilon_particle`). 
  - *Behavior*: If too low, you risk "sample impoverishment" where particles collapse into a tiny dot and fail to explore the space, eventually losing track of the true ball if the model is slightly wrong. If too high, the particles scatter too quickly, leading to a highly uncertain estimation that visually looks like a giant swarm.
- **Amount of particles**: 
  - *What it does*: The number of particles simulating the hypotheses.
  - *Behavior*: Higher means better estimation accuracy and robustness, but linearly increases CPU load and decreases FPS.
- **Environmental noise**: 
  - *What it does*: Adds random physics noise directly to the transition model (e.g., simulating random wind gusts).
  - *Behavior*: Makes the true physical trajectory of the balls unpredictable and harder for the filter to guess based purely on math.
- **Sensor noise**: 
  - *What it does*: The positional error added to the actual sensor readings. 
  - *Behavior*: If increased, the red 'X's on the plot will scatter further away from the true green balls, requiring the filter to rely less on individual measurements.
