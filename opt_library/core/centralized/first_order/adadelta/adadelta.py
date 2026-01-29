from typing import Any, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import DifferentiableProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class Adadelta(BaseAlgorithm):
    """Adadelta optimizer."""

    def __init__(self, rho: float = 0.9, epsilon: float = 1e-6):
        self.rho = rho
        self.epsilon = epsilon
        self.x: Optional[np.ndarray] = None
        self.ms_grad: Optional[np.ndarray] = None
        self.ms_delta: Optional[np.ndarray] = None

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
        self.ms_grad = np.zeros_like(self.x, dtype=float)
        self.ms_delta = np.zeros_like(self.x, dtype=float)

        while not stopping_condition or not stopping_condition.should_stop(
            {"x": self.x}
        ):
            self.step(problem=problem)
            if logger:
                logger.log({"x": self.x, "value": problem.evaluate(self.x)})

        return self.x, problem.evaluate(self.x)

    def step(self, problem: DifferentiableProblem, **kwargs: Any) -> None:
        """Perform a single step of the optimization algorithm."""
        if self.x is None or self.ms_grad is None or self.ms_delta is None:
            raise ValueError(
                "Optimizer not initialized. Call fit() before step()."
            )

        grad = problem.gradient(self.x)

        self.ms_grad = self.rho * self.ms_grad + (1 - self.rho) * grad**2

        rms_grad = np.sqrt(self.ms_grad + self.epsilon)
        rms_delta = np.sqrt(self.ms_delta + self.epsilon)

        delta_x = -(rms_delta / rms_grad) * grad

        self.ms_delta = self.rho * self.ms_delta + (1 - self.rho) * delta_x**2

        self.x = self.x + delta_x
