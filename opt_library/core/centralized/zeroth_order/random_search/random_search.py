import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm


class RandomSearch(BaseAlgorithm):
    """Random Search optimizer."""

    def __init__(self, num_samples=100, search_radius=1.0):
        super().__init__()
        self.num_samples = num_samples
        self.search_radius = search_radius

    def fit(self, problem, budget, seed=None):
        if seed is not None:
            np.random.seed(seed)

        best_solution = None
        best_fitness = float("inf")

        for _ in range(budget):
            solution = np.random.uniform(
                problem.lower_bound, problem.upper_bound, problem.dim
            )
            fitness = problem.evaluate(solution)

            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution

        return best_solution, best_fitness

    def step(self, problem, current_solution):
        # Generate a random direction and step
        direction = np.random.randn(problem.dim)
        direction /= np.linalg.norm(direction)

        # Create a new solution in the neighborhood
        new_solution = current_solution + self.search_radius * direction

        # Clip to bounds
        new_solution = np.clip(
            new_solution, problem.lower_bound, problem.upper_bound
        )

        return new_solution

    def simulate(self, problem, budget, seed=None):
        if seed is not None:
            np.random.seed(seed)

        history = []
        solution = np.random.uniform(
            problem.lower_bound, problem.upper_bound, problem.dim
        )

        for i in range(budget):
            fitness = problem.evaluate(solution)
            history.append((i, fitness))
            solution = self.step(problem, solution)

        return history
