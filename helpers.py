from enum import IntEnum
import random

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

    def reset(self):
        self.current_state = MRPState.C


def generate_episode():
    random_walk = RandomWalk()

    state = random_walk.current_state

    episode = []

    while not random_walk.in_terminal():
        next_state, reward = random_walk.step()
        episode.append((state, reward))
        state = next_state

    return episode, state