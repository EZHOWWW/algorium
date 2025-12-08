from typing import List

import numpy as np

from opt_library.core.common.base_problem import (
    ContinuousProblem,
    FiniteSumProblem,
)


class Sphere(ContinuousProblem):
    """Sphere function."""

    def __init__(
        self,
        dim: int = 10,
        lower_bound: float = -5.12,
        upper_bound: float = 5.12,
    ):
        super().__init__(bounds=[(lower_bound, upper_bound)] * dim)
        self._dimension = dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * x


class LinearRegressionProblem(FiniteSumProblem):
    """Linear Regression problem."""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self._dimension = X.shape[1]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    def evaluate(self, w: np.ndarray) -> float:
        error = self.X @ w - self.y
        return float(np.mean(error**2))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return 2 * self.X.T @ (self.X @ w - self.y) / len(self.y)

    def gradient_batch(self, w: np.ndarray, indices: List[int]) -> np.ndarray:
        X_batch, y_batch = self.X[indices], self.y[indices]
        return 2 * X_batch.T @ (X_batch @ w - y_batch) / len(y_batch)
