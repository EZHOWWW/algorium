# %%
import os
import sys

# This script can be run from the project root or from the notebooks directory.
# If we are in the notebooks directory, we need to add the project root to the path.
if os.path.basename(os.getcwd()) == "notebooks":
    sys.path.insert(0, os.path.abspath(".."))

# %%
import matplotlib.pyplot as plt
import numpy as np

from opt_library.core.centralized.first_order.gradient_descent.gradient_descent import (
    GradientDescent,
)
from opt_library.core.centralized.first_order.stochastic_gradient_descent.stochastic_gradient_descent import (
    StochasticGradientDescent,
)
from opt_library.core.centralized.zeroth_order.random_search.random_search import (
    RandomSearch,
)
from opt_library.core.centralized.zeroth_order.zeroth_order_sgd.zeroth_order_sgd import (
    ZOSGD,
)
from opt_library.core.common.base_problem import (
    ContinuousProblem,
    DifferentiableProblem,
)
from opt_library.simulator.base_logger import ListLogger
from opt_library.simulator.base_stopping_condition import (
    FevalsBudget,
    IterationBudget,
)
from opt_library.simulator.problems import LinearRegressionProblem

# ==============================================================================
# Test 1: Synthetic Differentiable Function (Rosenbrock)
# ==============================================================================


# %%
class RosenbrockProblem(ContinuousProblem):
    """Rosenbrock function problem."""

    def __init__(self, dim=2):
        super().__init__(bounds=[(-5, 5)] * dim)
        self._dimension = dim

    @property
    def dimension(self) -> int:
        return self._dimension

    def _evaluate(self, x: np.ndarray) -> float:
        """Evaluate the Rosenbrock function."""
        return float(
            np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
        )

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the Rosenbrock function."""
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        grad = np.zeros_like(x)
        grad[1:-1] = (
            200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
        )
        grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        grad[-1] = 200 * (x[-1] - x[-2] ** 2)
        return grad


def plot_convergence(logs, titles, use_fevals=False, xlim=None, ylim=None):
    """Plot convergence of algorithms."""
    plt.figure(figsize=(10, 6))
    for log, title in zip(logs, titles):
        if use_fevals:
            x_axis = [item["fevals"] for item in log]
            plt.xlabel("Number of Oracle Calls")
        else:
            x_axis = range(len(log))
            plt.xlabel("Iteration")
        values = [item["value"] for item in log]
        plt.plot(x_axis, values, label=title)

    plt.ylabel("Objective Function Value")
    plt.title("Algorithm Convergence")
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_gradient_norm(
    logs, titles, problem, use_fevals=False, xlim=None, ylim=None
):
    """Plot gradient norm of algorithms."""
    if not isinstance(problem, DifferentiableProblem):
        print("Skipping gradient norm plot for non-differentiable problem.")
        return
    plt.figure(figsize=(10, 6))
    for log, title in zip(logs, titles):
        if use_fevals:
            x_axis = [item["fevals"] for item in log]
            plt.xlabel("Number of Oracle Calls")
        else:
            x_axis = range(len(log))
            plt.xlabel("Iteration")
        norms = [np.linalg.norm(problem.gradient(item["x"])) for item in log]
        plt.plot(x_axis, norms, label=title)

    plt.ylabel("Gradient Norm")
    plt.title("Gradient Norm Convergence")
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.show()


# %%
print(
    "=============================================================================="
)
print("Test 1: Synthetic Differentiable Function (Rosenbrock)")
print(
    "=============================================================================="
)

# --- Problem and Starting Point ---
rosenbrock_problem = RosenbrockProblem(dim=2)
starting_point = np.array([0.0, 0.0])
iteration_budget = 100
fevals_budget = 2000

# --- First-Order Algorithms ---
print("\n--- Comparing First-Order Algorithms ---")
rosenbrock_problem.reset_fevals()
gd = GradientDescent(learning_rate=0.001)
gd_logger = ListLogger()
gd.fit(
    problem=rosenbrock_problem,
    starting_point=starting_point.copy(),
    stopping_condition=IterationBudget(iteration_budget),
    logger=gd_logger,
)

first_order_logs = [gd_logger.get_log()]
first_order_titles = ["Gradient Descent"]

plot_convergence(first_order_logs, first_order_titles)
plot_gradient_norm(first_order_logs, first_order_titles, rosenbrock_problem)


# --- Zeroth-Order Algorithms ---
print("\n--- Comparing Zeroth-Order Algorithms ---")
rosenbrock_problem.reset_fevals()
rs = RandomSearch(num_samples=10, search_radius=0.1)
rs_logger = ListLogger()
rs.fit(
    problem=rosenbrock_problem,
    starting_point=starting_point.copy(),
    stopping_condition=FevalsBudget(fevals_budget),
    logger=rs_logger,
)

rosenbrock_problem.reset_fevals()
zosgd = ZOSGD(learning_rate=0.001, mu=0.01, batch_size=4)
zosgd_logger = ListLogger()
zosgd.fit(
    problem=rosenbrock_problem,
    starting_point=starting_point.copy(),
    stopping_condition=FevalsBudget(fevals_budget),
    logger=zosgd_logger,
)

zeroth_order_logs = [rs_logger.get_log(), zosgd_logger.get_log()]
zeroth_order_titles = ["Random Search", "ZO-SGD"]

plot_convergence(
    zeroth_order_logs,
    zeroth_order_titles,
    use_fevals=True,
    xlim=(0, fevals_budget),
)

# %%
# ==============================================================================
# Test 2: Linear Regression
# ==============================================================================


print(
    "\n=============================================================================="
)
print("Test 2: Linear Regression")
print(
    "=============================================================================="
)

# --- Generate Synthetic Data ---
np.random.seed(42)
n_samples, n_features = 100, 5
X = np.random.rand(n_samples, n_features)
true_w = np.random.randn(n_features)
y = X @ true_w + 0.1 * np.random.randn(n_samples)

lin_reg_problem = LinearRegressionProblem(X, y)
starting_weights = np.zeros(n_features)
iteration_budget_lr = 200
fevals_budget_lr = 4000

# --- First-Order Algorithms ---
print("\n--- Comparing First-Order Algorithms (Linear Regression) ---")
lin_reg_problem.reset_fevals()
gd_lr = GradientDescent(learning_rate=0.1)
gd_lr_logger = ListLogger()
gd_lr.fit(
    problem=lin_reg_problem,
    starting_point=starting_weights.copy(),
    stopping_condition=IterationBudget(iteration_budget_lr),
    logger=gd_lr_logger,
)

lin_reg_problem.reset_fevals()
sgd_lr = StochasticGradientDescent(learning_rate=0.1, batch_size=16, epochs=10)
sgd_lr_logger = ListLogger()
sgd_lr.fit(
    problem=lin_reg_problem,
    starting_point=starting_weights.copy(),
    stopping_condition=IterationBudget(iteration_budget_lr),
    logger=sgd_lr_logger,
)

first_order_lr_logs = [gd_lr_logger.get_log(), sgd_lr_logger.get_log()]
first_order_lr_titles = ["Gradient Descent", "Stochastic Gradient Descent"]

plot_convergence(first_order_lr_logs, first_order_lr_titles, use_fevals=True)
plot_gradient_norm(
    first_order_lr_logs, first_order_lr_titles, lin_reg_problem, use_fevals=True
)


# --- Zeroth-Order Algorithms ---
print("\n--- Comparing Zeroth-Order Algorithms (Linear Regression) ---")
lin_reg_problem.reset_fevals()
rs_lr = RandomSearch(num_samples=20, search_radius=0.5)
rs_lr_logger = ListLogger()
rs_lr.fit(
    problem=lin_reg_problem,
    starting_point=starting_weights.copy(),
    stopping_condition=FevalsBudget(fevals_budget_lr),
    logger=rs_lr_logger,
)

lin_reg_problem.reset_fevals()
zosgd_lr = ZOSGD(learning_rate=0.1, mu=0.01, batch_size=4)
zosgd_lr_logger = ListLogger()
zosgd_lr.fit(
    problem=lin_reg_problem,
    starting_point=starting_weights.copy(),
    stopping_condition=FevalsBudget(fevals_budget_lr),
    logger=zosgd_lr_logger,
)

zeroth_order_lr_logs = [rs_lr_logger.get_log(), zosgd_lr_logger.get_log()]
zeroth_order_lr_titles = ["Random Search", "ZO-SGD"]

plot_convergence(
    zeroth_order_lr_logs,
    zeroth_order_lr_titles,
    use_fevals=True,
    xlim=(0, fevals_budget_lr),
)

print("\nComparison script finished.")
