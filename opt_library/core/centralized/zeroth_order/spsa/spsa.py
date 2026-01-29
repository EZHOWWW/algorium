from typing import Any, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import BaseProblem, ContinuousProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class SPSA(BaseAlgorithm):
    """Simultaneous Perturbation Stochastic Approximation (SPSA) optimizer."""

    def __init__(
        self,
        a: float = 0.05,
        c: float = 0.01,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
    ):
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A = A
        self.x: Optional[np.ndarray] = None
        self.y: Optional[float] = None
        self.k = 0

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
        self.k = 0

        if logger:
            logger.log(
                {"x": self.x, "value": self.y, "fevals": problem.get_fevals()}
            )

        while not stopping_condition or not stopping_condition.should_stop(
            {"x": self.x, "fevals": problem.get_fevals()}
        ):
            self.step(problem=problem)
            self.k += 1
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
        if self.x is None:
            raise ValueError(
                "Optimizer not initialized. Call fit() before step()."
            )

        # Calculate gain sequences
        ak = self.a / (self.A + self.k + 1) ** self.alpha
        ck = self.c / (self.k + 1) ** self.gamma

        # Generate random perturbation vector (Bernoulli distribution)
        delta = np.random.choice([-1, 1], size=problem.dimension)

        # Evaluate function at two points
        x_plus = self.x + ck * delta
        x_minus = self.x - ck * delta

        y_plus = problem.evaluate(x_plus)
        y_minus = problem.evaluate(x_minus)

        # Estimate gradient
        # Add a small epsilon to avoid division by zero
        grad_est = (y_plus - y_minus) / (2 * ck * delta + 1e-8)

        # Update x
        self.x = self.x - ak * grad_est

        # Clip to bounds if they are defined
        if isinstance(problem, ContinuousProblem) and problem.bounds:
            lower_bounds = np.array([b[0] for b in problem.bounds])
            upper_bounds = np.array([b[1] for b in problem.bounds])
            self.x = np.clip(self.x, lower_bounds, upper_bounds)

        self.y = problem.evaluate(self.x)
