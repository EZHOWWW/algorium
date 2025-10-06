import numpy as np


class Sphere:
    """Sphere function."""

    def __init__(self, dim=10, lower_bound=-5.12, upper_bound=5.12):
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, x):
        return np.sum(x**2)
