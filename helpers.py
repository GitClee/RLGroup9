from enum import IntEnum
import random
import numpy as np
import math

class MRPState(IntEnum):
    LEFT_TERM = 0
    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    RIGHT_TERM = 6

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class RandomWalk():
    def __init__(self):
        self.current_state = MRPState.C

    def step(self):
        if self.in_terminal():
            raise ValueError("Terminal state already reached")
        self.current_state = MRPState(self.current_state + random.choice([-1, 1]))
        reward = 1 if self.current_state == MRPState.RIGHT_TERM else 0
        return  self.current_state, reward

    def in_terminal(self):
        return self.current_state in (MRPState.LEFT_TERM, MRPState.RIGHT_TERM)

    def get_current_state(self):
        return self.current_state

    def reset(self):
        self.current_state = MRPState.C


def generate_episode():
    random_walk = RandomWalk()

    state = random_walk.get_current_state()

    episode = []

    while not random_walk.in_terminal():
        next_state, reward = random_walk.step()
        episode.append((state, reward))
        state = next_state

    return episode, state

# implements every visit Monte Carlo
class MonteCarlo:

    def __init__(self):
        self.state_values = {
            MRPState.A: 0.5,
            MRPState.B: 0.5,
            MRPState.C: 0.5,
            MRPState.D: 0.5,
            MRPState.E: 0.5,
            }
        self.alpha = 0.1
        self.gamma = 1 # no discounting


    def run_episodes(self, amount=1):
        for m in range(amount):
            episode, last_state = generate_episode()

            returns = self.get_returns(episode)

            for i in range(len(episode)):
                state = episode[i][0]

                self.state_values[state]  += self.alpha * (returns[i] - self.state_values[state])

    # calculate the returns for each state in episode
    def get_returns(self, episode):
        returns = []

        for i in range(len(episode)):
            val = 0
            for j in range(i, len(episode)):
                val += self.gamma**(j-i) * episode[j][1]

            returns.append(val)

        return returns

    # set step_size alpha
    def set_alpha(self, alpha):
        if not (0 < alpha <= 1): raise ValueError("Alpha must be in (0,1]")
        else: self.alpha = alpha

    # set gamma to enable discounting
    def set_gamma(self, gamma):
        if not (0 <= gamma <= 1): raise ValueError("Gamma must be in [0,1]")
        else: self.gamma = gamma

    def get_state_values(self):
        return self.state_values.copy()

    def reset(self):
        self.state_values =  {
            MRPState.A: 0.5,
            MRPState.B: 0.5,
            MRPState.C: 0.5,
            MRPState.D: 0.5,
            MRPState.E: 0.5,
            }

class OneStepTemporalDifference:

    def __init__(self):
        self.state_values = {
            MRPState.A: 0.5,
            MRPState.B: 0.5,
            MRPState.C: 0.5,
            MRPState.D: 0.5,
            MRPState.E: 0.5
        }
        self.alpha = 0.1
        self.gamma = 1
        self.random_walk = RandomWalk()

    def run_episodes(self, amount=1):

        for m in range(amount):

            current_state = MRPState.C

            while not self.random_walk.in_terminal():
                next_state, reward = self.random_walk.step()
                if self.random_walk.in_terminal():
                    error = reward - self.state_values[current_state]
                else:
                    error = reward + self.gamma*self.state_values[next_state] - self.state_values[current_state]
                self.state_values[current_state] += self.alpha * error
                current_state = next_state
            self.random_walk.reset()

    # set step_size alpha
    def set_alpha(self, alpha):
        if not (0 < alpha <= 1):
            raise ValueError("Alpha must be in (0,1]")
        else:
            self.alpha = alpha

    # set gamma to enable discounting
    def set_gamma(self, gamma):
        if not (0 <= gamma <= 1): raise ValueError("Gamma must be in [0,1]")
        else: self.gamma = gamma

    def get_state_values(self):
        return self.state_values.copy()

    def reset(self):
        self.state_values =  {
            MRPState.A: 0.5,
            MRPState.B: 0.5,
            MRPState.C: 0.5,
            MRPState.D: 0.5,
            MRPState.E: 0.5,
            }


def rms_error(real_values, obs_values):

    if len(real_values) != len(obs_values): raise ValueError("Lists must have same length")
    n = len(real_values)

    errors_squared = np.subtract(real_values, obs_values)**2

    rms = math.sqrt(sum(errors_squared)/n)

    return rms

def get_rms_over_episodes(method, max_episodes, runs):
    rms_errors = []
    real_values = (1/6, 2/6, 3/6, 4/6, 5/6)
    for i in range(runs):
        run = []
        for j in range(max_episodes):
            method.run_episodes()
            vals = method.get_state_values()
            run.append(rms_error(real_values, list(vals.values())))
        rms_errors.append(run)
        method.reset()

    rms_errors = np.mean(rms_errors, axis=0)

    return range(max_episodes), rms_errors














