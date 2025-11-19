# %%
import numpy as np

from opt_library.core.centralized.zeroth_order.random_search.random_search import (
    RandomSearch,
)
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition
from opt_library.simulator.problems import Sphere


class MaxIterations(BaseStoppingCondition):
    def __init__(self, max_iters: int):
        self.max_iters = max_iters
        self.iters = 0

    def should_stop(self, locals: dict) -> bool:
        self.iters += 1
        return self.iters >= self.max_iters


# %%
problem = Sphere(dim=10)
optimizer = RandomSearch(num_samples=100, search_radius=0.5)
starting_point = np.random.rand(10) * 10 - 5
stopping_condition = MaxIterations(max_iters=100)

solution, fitness = optimizer.fit(
    problem=problem,
    starting_point=starting_point,
    stopping_condition=stopping_condition,
)

print(f"Best solution found: {solution}")
print(f"Best fitness: {fitness}")
