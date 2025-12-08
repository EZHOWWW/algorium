from typing import Any, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import DifferentiableProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class Nesterov(BaseAlgorithm):
    """Nesterov Accelerated Gradient."""

    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.9):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.x: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None

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
        self.v = np.zeros_like(self.x)

        while not stopping_condition or not stopping_condition.should_stop(
            {"x": self.x}
        ):
            self.step(problem=problem)
            if logger:
                logger.log({"x": self.x, "value": problem.evaluate(self.x)})

        return self.x, problem.evaluate(self.x)

    def step(self, problem: DifferentiableProblem, **kwargs: Any) -> None:
        """Perform a single step of the optimization algorithm."""
        if self.x is None or self.v is None:
            raise ValueError(
                "Optimizer not initialized. Call fit() before step()."
            )

        x_lookahead = self.x - self.gamma * self.v
        grad_lookahead = problem.gradient(x_lookahead)

        self.v = self.gamma * self.v + self.learning_rate * grad_lookahead
        self.x = self.x - self.v
