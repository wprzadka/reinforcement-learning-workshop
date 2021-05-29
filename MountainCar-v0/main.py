import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    print(env.action_space)
    print(env.observation_space)

    for _ in range(1):
        done = False
        observation = env.reset()
        while not done:
            observation, reward, done, _ = env.step(np.random.choice((0, 1, 2)))
            env.render()

