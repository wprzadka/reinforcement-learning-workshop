import random

import numpy as np
from .Genetic import Genetic


class GeneticModelsSupervisor:

    def __init__(self, models_num: int):
        self.models_num = models_num
        self.models = np.array([Genetic() for _ in range(models_num)])

    def apply_random_mutations(self, models, mutation_probability: float = 0.1):
        for i, _ in enumerate(models):
            mut_matrix = np.random.rand(*models[i].action_matrix.shape) < mutation_probability
            for y, row in enumerate(mut_matrix):
                for x, mutate in enumerate(row):
                    if mutate:
                        models[i].action_matrix[y, x] = random.choice([0, 1, 2])
        return models

    def apply_mixing(self, models):
        children = np.empty_like(models)
        for i in range(0, len(models), 2):
            c1, c2 = self.mix_models(models[i], models[i + 1])
            children[i] = c1
            children[i + 1] = c2
        return children

    def mix_models(self, p1: Genetic, p2: Genetic, gen_fraction: float = 0.3):
        gen_matrix = np.random.rand(*p1.action_matrix.shape) < gen_fraction
        c1, c2 = Genetic(), Genetic()
        for y, row in enumerate(gen_matrix):
            for x, mix_gens in enumerate(row):
                if mix_gens:
                    c1.action_matrix[y, x] = p1.action_matrix[y, x]
                    c2.action_matrix[y, x] = p2.action_matrix[y, x]
                else:
                    c1.action_matrix[y, x] = p2.action_matrix[y, x]
                    c2.action_matrix[y, x] = p1.action_matrix[y, x]
        return c1, c2

    def apply_natural_selection(self, models, survivors_num: int, draw_rounds: int = 3):
        """
        survivors = np.empty(survivors_num, Genetic)
        for i, _ in enumerate(survivors):
            survivors[i] = np.random.choice(models)
            for _ in range(draw_rounds):
                subject = np.random.choice(models)
                survivors[i] = self.better_model(survivors[i], subject)
        return survivors
        """
        return np.array(sorted(models, key=lambda x: x.total_reward[1], reverse=True)[0:survivors_num + 1])

    def better_model(self, fst, snd):
        if fst.total_reward[0] == snd.total_reward[0]:
            return fst if fst.total_reward[1] > snd.total_reward[1] else snd
        return fst if fst.total_reward[0] > snd.total_reward[0] else snd

    def make_next_epoch(self):
        survivors = self.apply_natural_selection(self.models, self.models_num // 2)
        if survivors.size % 2 == 1:
            survivors.resize((survivors.size - 1,))
        children = self.apply_mixing(survivors)
        self.models = self.apply_random_mutations(np.concatenate((survivors, children)))
        for model in self.models:
            model.clear_reward()
        assert len(self.models) == self.models_num
