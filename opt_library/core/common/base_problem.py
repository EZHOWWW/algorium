from abc import ABC, abstractmethod

import numpy as np


class BaseProblem(ABC):
    """Abstract base class for optimization problems."""

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective function at a given point."""
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the objective function at a given point."""
        pass
