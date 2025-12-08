from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BaseProblem(ABC):
    """
    Базовый класс для ЛЮБОЙ задачи оптимизации.
    Обязан уметь считать значение.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Размерность пространства (количество параметров)."""
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Значение целевой функции f(x)."""
        pass


class DifferentiableProblem(BaseProblem):
    """Маркерный класс для задач, у которых точно есть градиент."""

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Полный градиент функции в точке x.
        Для ML задач это градиент по ВСЕМУ датасету.
        """
        pass


# --- Ветка для ML (Data-Driven) ---


class FiniteSumProblem(DifferentiableProblem):
    """
    Задачи, представимые как сумма функций потерь на примерах (ML).
    Поддерживает батчинг.
    """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        pass

    @abstractmethod
    def gradient_batch(self, x: np.ndarray, indices: List[int]) -> np.ndarray:
        """Градиент только по выбранным индексам."""
        pass


# --- Ветка для Непрерывной оптимизации (Math) ---


class ContinuousProblem(DifferentiableProblem):
    """
    Классическая математическая оптимизация (например, f(x) = x^2).
    Может иметь границы (Bounds).
    """

    def __init__(self, bounds: Optional[List[Tuple[float, float]]] = None):
        self.bounds = bounds  # [(min, max), ...]

    # Для непрерывных задач gradient_batch не имеет смысла в контексте данных,
    # поэтому мы его не реализуем.
