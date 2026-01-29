from typing import Any, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import FiniteSumProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class StochasticGradientDescent(BaseAlgorithm):
    """Stochastic Gradient Descent algorithm."""

    def __init__(
        self, learning_rate: float = 0.1, batch_size: int = 1, epochs: int = 10
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.x: Optional[np.ndarray] = None
        self.y: Optional[float] = None

    def fit(
        self,
        problem: FiniteSumProblem,
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
                {
                    "x": self.x,
                    "value": self.y,
                    "fevals": problem.get_fevals(),
                    "epoch": 0,
                }
            )

        for epoch in range(self.epochs):
            if stopping_condition and stopping_condition.should_stop(
                {"x": self.x, "fevals": problem.get_fevals()}
            ):
                break
            self.step(problem=problem)

            if logger:
                logger.log(
                    {
                        "x": self.x,
                        "value": self.y,
                        "fevals": problem.get_fevals(),
                        "epoch": epoch + 1,
                    }
                )

        return self.x, self.y

    def step(self, problem: FiniteSumProblem, **kwargs: Any) -> None:
        """Perform a single step of the optimization algorithm."""

        n_samples = problem.n_samples

        indices = np.random.choice(n_samples, self.batch_size, replace=False)

        grad = problem.gradient_batch(self.x, indices=indices)

        self.x = self.x - self.learning_rate * grad
        self.y = problem.evaluate(self.x)
