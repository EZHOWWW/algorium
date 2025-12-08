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


class Beale(ContinuousProblem):
    """Beale function. A 2D-only problem."""

    def __init__(self):
        super().__init__(bounds=[(-4.5, 4.5)] * 2)
        self.min_value = 0.0
        self.min_point = np.array([3.0, 0.5])

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, x: np.ndarray) -> float:
        if x.shape[0] != 2:
            raise ValueError("Beale function is defined for 2 dimensions only.")
        x_val, y_val = x[0], x[1]
        term1 = (1.5 - x_val + x_val * y_val) ** 2
        term2 = (2.25 - x_val + x_val * y_val**2) ** 2
        term3 = (2.625 - x_val + x_val * y_val**3) ** 2
        return float(term1 + term2 + term3)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != 2:
            raise ValueError("Beale function is defined for 2 dimensions only.")
        x_val, y_val = x[0], x[1]
        grad = np.zeros(2)

        # Common terms
        t1 = 1.5 - x_val + x_val * y_val
        t2 = 2.25 - x_val + x_val * y_val**2
        t3 = 2.625 - x_val + x_val * y_val**3

        # Partial derivative with respect to x
        grad[0] = (
            2 * t1 * (y_val - 1)
            + 2 * t2 * (y_val**2 - 1)
            + 2 * t3 * (y_val**3 - 1)
        )

        # Partial derivative with respect to y
        grad[1] = (
            2 * t1 * x_val
            + 2 * t2 * (2 * x_val * y_val)
            + 2 * t3 * (3 * x_val * y_val**2)
        )

        return grad


class Ackley(ContinuousProblem):
    """Ackley function. N-dimensional problem with many local minima."""

    def __init__(self, dim: int = 2, a=20, b=0.2, c=2 * np.pi):
        super().__init__(bounds=[(-32.768, 32.768)] * dim)
        self._dimension = dim
        self.a = a
        self.b = b
        self.c = c
        self.min_value = 0.0
        self.min_point = np.zeros(dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    def evaluate(self, x: np.ndarray) -> float:
        sum_sq_term = -self.b * np.sqrt(np.sum(x**2) / self.dimension)
        cos_term = np.sum(np.cos(self.c * x)) / self.dimension

        return float(
            -self.a * np.exp(sum_sq_term)
            - np.exp(cos_term)
            + self.a
            + np.exp(1)
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        sum_sq = np.sum(x**2)
        sqrt_sum_sq = np.sqrt(sum_sq)

        if sqrt_sum_sq == 0:
            return np.zeros_like(x)

        common_term = (
            self.a
            * self.b
            / (sqrt_sum_sq * self.dimension)
            * np.exp(-self.b * sqrt_sum_sq / np.sqrt(self.dimension))
        )

        grad = common_term * x + (self.c / self.dimension) * np.sin(
            self.c * x
        ) * np.exp(np.sum(np.cos(self.c * x)) / self.dimension)
        return grad
