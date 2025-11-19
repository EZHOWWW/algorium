from typing import Any, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import BaseProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class RandomSearch(BaseAlgorithm):
    """Random Search optimizer."""

    def __init__(self, num_samples: int = 100, search_radius: float = 1.0):
        self.num_samples = num_samples
        self.search_radius = search_radius
        self.x: Optional[np.ndarray] = None
        self.y: Optional[float] = None

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

        if logger:
            logger.log({"x": self.x, "value": self.y})

        while not stopping_condition or not stopping_condition.should_stop(
            {"x": self.x}
        ):
            self.step(problem=problem)
            if logger:
                logger.log({"x": self.x, "value": self.y})

        return self.x, self.y

    def step(self, problem: BaseProblem, **kwargs: Any) -> None:
        """Perform a single step of the optimization algorithm."""
        if self.x is None or self.y is None:
            raise ValueError(
                "Optimizer not initialized. Call fit() before step()."
            )

        # Generate candidate solutions in the neighborhood
        candidates = [
            self.x + self.search_radius * np.random.randn(problem.dim)
            for _ in range(self.num_samples)
        ]

        # Clip to bounds if they are defined
        if hasattr(problem, "lower_bound") and hasattr(problem, "upper_bound"):
            candidates = [
                np.clip(c, problem.lower_bound, problem.upper_bound)
                for c in candidates
            ]

        # Evaluate candidates
        candidate_evals = [problem.evaluate(c) for c in candidates]

        # Find the best candidate
        best_candidate_idx = np.argmin(candidate_evals)
        best_candidate_eval = candidate_evals[best_candidate_idx]

        # If a better solution is found, update the current solution
        if best_candidate_eval < self.y:
            self.x = candidates[best_candidate_idx]
            self.y = best_candidate_eval
