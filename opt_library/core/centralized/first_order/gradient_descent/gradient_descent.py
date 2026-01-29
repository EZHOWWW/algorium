from typing import Any, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import DifferentiableProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class GradientDescent(BaseAlgorithm):
    """Basic Gradient Descent algorithm."""

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.x: Optional[np.ndarray] = None
        self.y: Optional[float] = None

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
        self.y = problem.evaluate(self.x)

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

    def step(self, problem: DifferentiableProblem, **kwargs: Any) -> None:
        """Perform a single step of the optimization algorithm."""
        grad = problem.gradient(self.x)
        self.x = self.x - self.learning_rate * grad
        self.y = problem.evaluate(self.x)
