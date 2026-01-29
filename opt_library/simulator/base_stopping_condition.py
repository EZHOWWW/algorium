from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStoppingCondition(ABC):
    """Abstract base class for stopping conditions."""

    @abstractmethod
    def should_stop(self, context: Dict[str, Any]) -> bool:
        """Determine if the optimization should stop."""
        pass


class IterationBudget(BaseStoppingCondition):
    """Stop when a budget of iterations is exhausted."""

    def __init__(self, budget: int):
        self.budget = budget
        self.iterations = 0

    def should_stop(self, context: Dict[str, Any]) -> bool:
        self.iterations += 1
        return self.iterations >= self.budget


class FevalsBudget(BaseStoppingCondition):
    """Stop when a budget of function evaluations is exhausted."""

    def __init__(self, budget: int):
        self.budget = budget

    def should_stop(self, context: Dict[str, Any]) -> bool:
        fevals = context.get("fevals")
        if fevals is None:
            return False
        return fevals >= self.budget
