import numpy as np


class Genetic:

    # observation -> (position [-1.2, 0.6], velocity [-0.7, 0.7])
    # action -> (move [0 - left, 1 - none, 2 - right])

    def __init__(self):
        self.total_reward = np.array((-999, -999), float)
        self.action_matrix = np.ones((19, 15), int)

    def update_reward(self, reward, observation):
        self.total_reward[0] += reward
        self.total_reward[1] = max(self.total_reward[1], observation[0])

    def make_action(self, observation):
        idx = self.discretize_observation(observation)
        return self.action_matrix[idx[0], idx[1]]

    def discretize_observation(self, observation):
        # get position bucket index (scaling range to [0, 18])
        position = int((observation[0] + 1.2) * 10)
        # get velocity bucket index (range [0, 14])
        velocity = int((observation[1] + 0.7) * 10)
        return [position, velocity]

    def clear_reward(self):
        self.total_reward = np.array((0, -1.2), float)