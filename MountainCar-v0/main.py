import gym
import numpy as np
from models.GeneticModelsSupervisor import GeneticModelsSupervisor

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    epoch_limit = 50
    number_of_models = 24

    print(env.action_space)
    print(env.observation_space)
    supervisor = GeneticModelsSupervisor(number_of_models)

    for epoch_num in range(epoch_limit):
        max_total_reward = -200
        for model in supervisor.models:
            done = False
            observation = env.reset()
            while not done:
                observation, reward, done, _ = env.step(model.make_action(observation))
                model.update_reward(reward, observation)
                if epoch_num == epoch_limit - 1:
                    env.render()
            max_total_reward = max(max_total_reward, model.total_reward[0])
        print(f'epoch {epoch_num}/{epoch_limit} max_reward: {max_total_reward}')
        supervisor.make_next_epoch()
