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
