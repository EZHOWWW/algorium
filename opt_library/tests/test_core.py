from opt_library.core.centralized.zeroth_order.random_search.random_search import (
    RandomSearch,
)
from opt_library.simulator.problems import Sphere


def test_random_search_fit():
    problem = Sphere(dim=2)
    optimizer = RandomSearch(num_samples=1000)
    solution, fitness = optimizer.fit(problem, budget=1000, seed=42)
    assert fitness < 1.0, f"Fitness should be less than 1.0, but got {fitness}"


def test_random_search_step():
    problem = Sphere(dim=2)
    optimizer = RandomSearch()
    initial_solution = [1.0, 1.0]
    new_solution = optimizer.step(problem, initial_solution)
    assert new_solution is not None


def test_random_search_simulate():
    problem = Sphere(dim=2)
    optimizer = RandomSearch()
    history = optimizer.simulate(problem, budget=100, seed=42)
    assert len(history) == 100
