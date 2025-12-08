# Stochastic Gradient Descent (SGD) Implementation Plan

## 1. Objective

Implement the Stochastic Gradient Descent (SGD) optimization algorithm within the existing `opt_library` framework. This algorithm is a first-order, centralized method suitable for large-scale and online learning problems.

## 2. Key Idea

Stochastic Gradient Descent is an iterative optimization algorithm used for minimizing an objective function. Unlike standard Gradient Descent (GD), which computes the gradient of the objective function using the entire dataset, SGD approximates the gradient by using a small, randomly selected subset of the data called a "mini-batch."

The update rule for a parameter vector `x` is:
`x = x - learning_rate * stochastic_gradient(x, mini_batch)`

This approach offers several advantages:
- **Scalability**: Each iteration is computationally cheaper, making it suitable for very large datasets.
- **Faster Convergence**: Although individual updates are noisy, SGD often converges faster in practice because it can perform many more updates in the same amount of time as one GD update.
- **Regularization Effect**: The noise in the gradient updates can help the algorithm escape shallow local minima.

## 3. Implementation Plan

1.  **Directory Structure**: Create the directory `opt_library/core/centralized/first_order/stochastic_gradient_descent/`.

2.  **Extend Problem Interface**: To properly support stochastic methods, the `BaseProblem` class will be updated.
    *   Add a new, optional method `stochastic_gradient(self, x: np.ndarray, batch: Any) -> np.ndarray` to `opt_library/core/common/base_problem.py`. This will allow problems to define how to compute a gradient on a mini-batch.

3.  **Update Existing Problems**:
    *   Modify `RosenbrockProblem` in `notebooks/02_algorithm_comparison_repl.py` to implement `stochastic_gradient`. Since it's a deterministic problem, this method will simply return the full gradient, ignoring the batch.
    *   Modify `LinearRegressionProblem` in `notebooks/02_algorithm_comparison_repl.py` to implement `stochastic_gradient`. This method will compute the gradient based on the provided mini-batch of data `(X_batch, y_batch)`.

4.  **Algorithm Implementation (`stochastic_gradient_descent.py`)**:
    *   Create a `StochasticGradientDescent` class that inherits from `BaseAlgorithm`.
    *   The `__init__` method will accept `learning_rate`, `batch_size`, and `epochs`.
    *   The `fit` method will be the main entry point. It will:
        *   Handle the main optimization loop, which iterates for a specified number of `epochs`.
        *   In each epoch, it will shuffle the dataset (assuming the `problem` object has `X` and `y` attributes).
        *   It will then iterate through the shuffled data in mini-batches.
        *   For each mini-batch, it will call the `step` method.
        *   It will check the `stopping_condition` after each step (mini-batch update).
    *   The `step` method will perform a single optimization step. It will:
        *   Call the `problem.stochastic_gradient(self.x, batch)` method to get the gradient for the current mini-batch.
        *   Update the parameters `self.x` using the SGD update rule.

5.  **Metadata and Documentation**:
    *   Create `stochastic_gradient_descent.yaml` with appropriate tags (e.g., `first_order`, `noisy_oracle`).
    *   Create a `README.md` file explaining the algorithm and its parameters.

6.  **Testing**:
    *   Add a new test case to `tests/test_core.py` to verify that `StochasticGradientDescent` can successfully optimize the `LinearRegressionProblem`.

7.  **Usage Example**:
    *   Update the comparison script `notebooks/02_algorithm_comparison_repl.py`.
    *   Instantiate and run `StochasticGradientDescent` on the `LinearRegressionProblem`.
    *   Add its results to the comparison plots to visualize its convergence behavior against standard Gradient Descent.
