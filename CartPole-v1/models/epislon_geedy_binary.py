import numpy as np

class EpsilonGreedy:

    def __init__(self, epsilon):
        self.last_reward = 0
        self.average_reward = np.zeros((2, 2))
        self.epsilon = epsilon
        self.current_epoch_action = np.ones((2, 2))

    def make_action(self, env_state):
        state = self.normalize_state(env_state)
        if np.random.uniform() > self.epsilon:
            action = np.argmax(self.average_reward[state])
        else:
            action = np.random.choice((0, 1))

        self.current_epoch_action[state][action] += 1
        return action

    def normalize_state(self, env_state):
        return 1 if np.sign(env_state[3]) == 1 else 0

    def update_rewards(self, current_reward):
        diff = current_reward - self.last_reward
        self.average_reward += diff / self.current_epoch_action
