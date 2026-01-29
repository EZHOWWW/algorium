from typing import Any, Optional, Tuple

import numpy as np

from opt_library.core.common.base_algorithm import BaseAlgorithm
from opt_library.core.common.base_problem import BaseProblem, ContinuousProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class MeZO(BaseAlgorithm):
    """Memory-efficient Zeroth-Order (MeZO) optimizer.

    This implementation is based on the paper "MeZO: Memory-Efficient
    Zeroth-Order Optimization for Fine-tuning Large Language Models".
    It uses in-place updates and re-generates random perturbations from a
    seed to achieve O(1) memory complexity with respect to model parameters.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 0.01,
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
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
        self.x = starting_point.copy()  # Work on a copy
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

        # Generate a seed for this step to ensure z is the same for all ops
        step_seed = np.random.randint(0, 2**32 - 1)
        rng = np.random.default_rng(step_seed)
        z = rng.standard_normal(size=problem.dimension)

        # Step 1: Forward evaluation with positive perturbation
        # In-place update: x_plus = self.x + self.epsilon * z
        np.add(self.x, self.epsilon * z, out=self.x)
        y_plus = problem.evaluate(self.x)

        # Step 2: Forward evaluation with negative perturbation
        # In-place update to move from (x + ez) to (x - ez)
        # self.x = (self.x - self.epsilon * z) - self.epsilon * z
        np.subtract(self.x, 2 * self.epsilon * z, out=self.x)
        y_minus = problem.evaluate(self.x)

        # Step 3: Restore original parameters and apply the update
        # Estimate projected gradient (scalar)
        denominator = 2 * self.epsilon
        if denominator == 0:
            grad_proj = 0.0
        else:
            grad_proj = (y_plus - y_minus) / denominator

        # In-place update to move from (x - ez) to the final updated position
        # self.x = (self.x + ez) - lr * grad_proj * z
        # self.x = self.x + (e - lr * grad_proj) * z
        update_magnitude = self.epsilon - self.learning_rate * grad_proj
        np.add(self.x, update_magnitude * z, out=self.x)

        # Clip to bounds if they are defined
        if isinstance(problem, ContinuousProblem) and problem.bounds:
            lower_bounds = np.array([b[0] for b in problem.bounds])
            upper_bounds = np.array([b[1] for b in problem.bounds])
            np.clip(self.x, lower_bounds, upper_bounds, out=self.x)

        # Final evaluation of the new point
        self.y = problem.evaluate(self.x)
