from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLogger(ABC):
    """Abstract base class for loggers."""

    @abstractmethod
    def log(self, data: Dict[str, Any]) -> None:
        """Log data."""
        pass

    @abstractmethod
    def get_log(self) -> List[Dict[str, Any]]:
        """Get the entire log."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the logger."""
        pass


class ListLogger(BaseLogger):
    """A simple logger that stores data in a list."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def log(self, data: Dict[str, Any]) -> None:
        self.history.append(data)

    def get_log(self) -> List[Dict[str, Any]]:
        return self.history

    def reset(self) -> None:
        self.history = []
