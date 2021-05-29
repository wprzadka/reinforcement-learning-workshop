import numpy as np
from Genetic import Genetic


class GeneticModelsSupervisor:

    def __init__(self, models_num: int):
        self.models = np.array([Genetic() for _ in range(models_num)])

    def apply_random_mutations(self):
        pass

    def apply_natural_selection(self):
        pass

    def make_next_epoch(self):
        pass
