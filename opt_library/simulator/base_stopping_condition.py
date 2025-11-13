from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStoppingCondition(ABC):
    """Abstract base class for stopping conditions."""

    @abstractmethod
    def should_stop(self, context: Dict[str, Any]) -> bool:
        """Determine if the optimization should stop."""
        pass


class BudgetStoppingCondition(BaseStoppingCondition):
    """Stop when a budget is exhausted."""

    def __init__(self, budget: int):
        self.budget = budget
        self.iterations = 0

    def should_stop(self, context: Dict[str, Any]) -> bool:
        self.iterations += 1
        return self.iterations >= self.budget
