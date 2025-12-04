import numpy as np

from opt_library.core.common.base_problem import BaseProblem


class Sphere(BaseProblem):
    """Sphere function."""

    def __init__(
        self,
        dim: int = 10,
        lower_bound: float = -5.12,
        upper_bound: float = 5.12,
    ):
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * x


class LinearRegressionProblem(BaseProblem):
    """Linear Regression problem."""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.dim = X.shape[1]

    def evaluate(self, w: np.ndarray) -> float:
        error = self.X @ w - self.y
        return float(np.mean(error**2))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        return 2 * self.X.T @ (self.X @ w - self.y) / len(self.y)

    def stochastic_gradient(self, w: np.ndarray, batch: tuple) -> np.ndarray:
        X_batch, y_batch = batch
        return 2 * X_batch.T @ (X_batch @ w - y_batch) / len(y_batch)
