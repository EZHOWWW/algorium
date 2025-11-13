from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseNoise(ABC):
    """Abstract base class for noise models."""

    @abstractmethod
    def apply(self, value: Any) -> Any:
        """Apply noise to a value."""
        pass


class GaussianNoise(BaseNoise):
    """Gaussian noise model."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def apply(self, value: np.ndarray) -> np.ndarray:
        return value + np.random.normal(self.mean, self.std, value.shape)
