from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from opt_library.core.common.base_problem import BaseProblem
from opt_library.simulator.base_logger import BaseLogger
from opt_library.simulator.base_network import BaseNetwork
from opt_library.simulator.base_noise import BaseNoise
from opt_library.simulator.base_stopping_condition import BaseStoppingCondition


class BaseAlgorithm(ABC):
    """Abstract base class for all optimization algorithms."""

    def __init__(self, **kwargs: Any):
        pass

    @abstractmethod
    def fit(
        self,
        problem: BaseProblem,
        stopping_condition: Optional[BaseStoppingCondition] = None,
        logger: Optional[BaseLogger] = None,
        network: Optional[BaseNetwork] = None,
        noise: Optional[BaseNoise] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Fit the algorithm to a given problem."""
        pass

    @abstractmethod
    def step(self, **kwargs: Any) -> None:
        """Perform a single step of the optimization algorithm."""
        pass

    def simulate(
        self,
        problem: BaseProblem,
        stopping_condition: Optional[BaseStoppingCondition] = None,
        logger: Optional[BaseLogger] = None,
        network: Optional[BaseNetwork] = None,
        noise: Optional[BaseNoise] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """Simulate the algorithm's performance on a problem."""
        if logger:
            logger.reset()

        solution = None
        while not stopping_condition or not stopping_condition.should_stop(
            {"solution": solution}
        ):
            solution = self.step(problem=problem, **kwargs)
            if logger:
                logger.log({"solution": solution})

        return logger.get_log() if logger else []
