from typing import Any, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import DifferentiableProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class Adam(BaseAlgorithm):
    """Adam optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.x: Optional[np.ndarray] = None
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0

    def fit(
        self,
        problem: DifferentiableProblem,
        starting_point: np.ndarray,
        stopping_condition: Optional[BaseStoppingCondition] = None,
        logger: Optional[BaseLogger] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Fit the algorithm to a given problem."""
        self.x = starting_point
        self.m = np.zeros_like(self.x, dtype=float)
        self.v = np.zeros_like(self.x, dtype=float)
        self.t = 0

        while not stopping_condition or not stopping_condition.should_stop(
            {"x": self.x, "fevals": problem.get_fevals()}
        ):
            self.step(problem=problem)
            if logger:
                logger.log(
                    {
                        "x": self.x,
                        "value": problem.evaluate(self.x),
                        "fevals": problem.get_fevals(),
                    }
                )

        return self.x, problem.evaluate(self.x)

    def step(self, problem: DifferentiableProblem, **kwargs: Any) -> None:
        """Perform a single step of the optimization algorithm."""
        if self.x is None or self.m is None or self.v is None:
            raise ValueError(
                "Optimizer not initialized. Call fit() before step()."
            )

        self.t += 1
        grad = problem.gradient(self.x)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        self.x = (
            self.x
            - (self.learning_rate / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        )
