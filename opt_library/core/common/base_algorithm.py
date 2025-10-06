from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """Abstract base class for all optimization algorithms."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, problem, **kwargs):
        """Fit the algorithm to a given problem."""
        pass

    @abstractmethod
    def step(self, **kwargs):
        """Perform a single step of the optimization algorithm."""
        pass

    def simulate(self, **kwargs):
        """Simulate the algorithm's performance on a problem."""
        pass
