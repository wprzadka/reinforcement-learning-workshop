import gym
from models.epislon_geedy_binary import EpsilonGreedy
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    print(f'actions: {env.action_space}')
    print(f'observations: {env.observation_space}')

    model = EpsilonGreedy(epsilon=0.3)

    episodes_rewards = []
    episodes_num = 10_000
    for _ in range(episodes_num):  # episodes
        done = False
        current_total_reward = 0
        observation = env.reset()
        t = 0
        while not done:
            env.render()
            observation, reward, done, _ = env.step(model.make_action(observation))
            current_total_reward += reward
            t += 1
        model.update_rewards(current_total_reward)
        episodes_rewards.append(current_total_reward)
        print(f'ends after {t} with total reward of {current_total_reward}')
    env.close()
    print(np.max(episodes_rewards))