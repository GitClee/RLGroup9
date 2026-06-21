# Multi-Ball Particle Filter

This project implements a Joint State Particle Filter for tracking $n \ge 1$ indistinguishable balls simultaneously, handling severe observation noise and complete sensor dropouts.

## Dependencies

The project uses standard scientific Python libraries. Make sure you have them installed:

```bash
pip install numpy scipy scikit-learn matplotlib
```

## Running the Simulation

Simply execute the main script:

```bash
python main.py
```

An interactive Matplotlib window will open, showing a live animation of:
- **Blue circles**: True ball positions
- **Red crosses**: Noisy observations (when sensors are active)
- **Faint gray cloud**: Particle positions representing the belief density
- **Green squares**: Estimated ball positions extracted via K-Means clustering

## Code Structure

- `simulation.py`: Handles the ground-truth physical trajectory of $n$ balls (gravity, bouncing) and generates noisy, shuffled observations with optional dropouts.
- `particle_filter.py`: The core algorithm.
  - **Predict**: Applies physical motion and process noise.
  - **Update**: Uses the **Hungarian Algorithm** (Linear Sum Assignment) to compute the optimal matching between particle predictions and indistinguishable observations, producing a robust likelihood.
  - **Estimate**: Uses **K-Means clustering** to naturally discover the $n$ highest density regions from the joint particle distribution, avoiding the "label switching" problem that occurs with naive averaging.
- `visualization.py`: A `matplotlib` helper that renders the live interactive tracking plot.
- `main.py`: Ties everything together.

## Changing Parameters

You can easily configure the simulation in `main.py`:
- `n_balls`: Change the number of balls flying simultaneously.
- `dropout_prob`: Change how often the sensors fail completely (e.g., `0.15` means a 15% chance of zero observations in a frame).
- `sim_noise_std`: Adjust the amount of Gaussian noise added to the positions.

## Interview Prep Details
- **Indistinguishability**: Addressed by shuffling observations and scoring particles using the minimal assignment cost (Hungarian Algorithm).
- **Label Switching & Multimodality**: Addressed by abandoning standard state-averaging. Instead, we flatten all positional predictions across the particle swarm and cluster them into $n$ centers.
- **Data Dropouts**: When zero observations arrive, the filter simply predicts forward and skips the likelihood update, properly dispersing the particle cloud over time.
