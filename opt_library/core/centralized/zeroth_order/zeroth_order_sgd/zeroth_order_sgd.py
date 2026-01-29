from typing import Any, Callable, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import BaseProblem, ContinuousProblem
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
        self.x: Optional[np.ndarray] = None
        self.y: Optional[float] = None
        self.n_dim = None

    def _sample_unit_gaussian(self) -> np.ndarray:
        """Generate a random vector from N(0, I)."""
        return np.random.standard_normal(self.n_dim)

    def _estimate_gradient(
        self,
        func: Callable[[np.ndarray], float],
        x: np.ndarray,
        y: float,
        u: np.ndarray,
    ) -> np.ndarray:
        """
        Implement Equation (3.12) from the paper:
        G_mu = [(F(x + mu*u) - F(x)) / mu] * u
        """
        val_perturbed = func(x + self.mu * u)
        diff = (val_perturbed - y) / self.mu
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
        self.y = problem.evaluate(self.x)
        self.n_dim = problem.dimension

        if logger:
            logger.log(
                {"x": self.x, "value": self.y, "fevals": problem.get_fevals()}
            )

        while not stopping_condition or not stopping_condition.should_stop(
            {"x": self.x, "fevals": problem.get_fevals()}
        ):
            self.step(problem=problem)
            if logger:
                logger.log(
                    {
                        "x": self.x,
                        "value": self.y,
                        "fevals": problem.get_fevals(),
                    }
                )

        return self.x, self.y

    def step(self, problem: BaseProblem, **kwargs: Any) -> None:
        """Perform a single step of the optimization algorithm."""
        grad_estimate = np.zeros(self.n_dim)
        for _ in range(self.batch_size):
            u_k = self._sample_unit_gaussian()
            grad_estimate += self._estimate_gradient(
                problem.evaluate, self.x, self.y, u_k
            )

        grad_estimate /= self.batch_size
        self.x = self.x - self.learning_rate * grad_estimate

        # Clip to bounds if they are defined
        if isinstance(problem, ContinuousProblem) and problem.bounds:
            lower_bounds = np.array([b[0] for b in problem.bounds])
            upper_bounds = np.array([b[1] for b in problem.bounds])
            self.x = np.clip(self.x, lower_bounds, upper_bounds)

        self.y = problem.evaluate(self.x)
