from typing import Any, Callable, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import BaseProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class ZOSGD(BaseAlgorithm):
    """Zeroth-Order Stochastic Gradient Descent (ZO-SGD), also known as RSGF."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        mu: float = 1e-3,
        batch_size: int = 1,
    ):
        self.learning_rate = learning_rate
        self.mu = mu
        self.batch_size = batch_size
        self.x = None
        self.n_dim = None

    def _sample_unit_gaussian(self) -> np.ndarray:
        """Generate a random vector from N(0, I)."""
        return np.random.standard_normal(self.n_dim)

    def _estimate_gradient(
        self, func: Callable[[np.ndarray], float], x: np.ndarray, u: np.ndarray
    ) -> np.ndarray:
        """
        Implement Equation (3.12) from the paper:
        G_mu = [(F(x + mu*u) - F(x)) / mu] * u
        """
        val_perturbed = func(x + self.mu * u)
        val_current = func(x)
        diff = (val_perturbed - val_current) / self.mu
        grad_estimate = diff * u
        return grad_estimate

    def fit(
        self,
        problem: BaseProblem,
        starting_point: np.ndarray,
        stopping_condition: Optional[BaseStoppingCondition] = None,
        logger: Optional[BaseLogger] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Fit the algorithm to a given problem."""
        self.x = starting_point
        self.n_dim = len(starting_point)

        while not stopping_condition or not stopping_condition.should_stop(
            {"x": self.x}
        ):
            self.step(problem=problem)
            if logger:
                logger.log({"x": self.x, "value": problem.evaluate(self.x)})

        return self.x, problem.evaluate(self.x)

    def step(self, problem: BaseProblem, **kwargs: Any) -> None:
        """Perform a single step of the optimization algorithm."""
        grad_estimate = np.zeros(self.n_dim)
        for _ in range(self.batch_size):
            u_k = self._sample_unit_gaussian()
            grad_estimate += self._estimate_gradient(
                problem.evaluate, self.x, u_k
            )

        grad_estimate /= self.batch_size
        self.x = self.x - self.learning_rate * grad_estimate
