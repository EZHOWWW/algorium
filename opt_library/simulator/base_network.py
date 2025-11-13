from abc import ABC, abstractmethod
from typing import List


class BaseNetwork(ABC):
    """Abstract base class for network topologies."""

    @abstractmethod
    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get the neighbors of a given agent."""
        pass
