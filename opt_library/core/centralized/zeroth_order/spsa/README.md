# Simultaneous Perturbation Stochastic Approximation (SPSA)

This directory contains the implementation of the SPSA algorithm.

## Description

SPSA is a stochastic approximation algorithm for finding a local minimum of a function. It is particularly useful for high-dimensional problems. The key idea of SPSA is to approximate the gradient of the function using only two function evaluations, regardless of the dimension of the problem.

The SPSA update rule is:
$$\theta\_{k+1} = \theta_k - a_k \hat{g}\_k(\theta_k)$$

The gradient estimate $$ \hat{g}\_k(\theta_k) $$ is computed as:
$$\hat{g}\_k(\theta_k) = \frac{f(\theta_k + c_k \Delta_k) - f(\theta_k - c_k \Delta_k)}{2 c_k \Delta_k}$$

where $\Delta_k$ is a random perturbation vector, and $a_k$ and $c_k$ are gain sequences.

## Files

- `spsa.py`: The main implementation of the SPSA algorithm.
- `spsa.yaml`: Default parameters for the SPSA algorithm.
- `README.md`: This file.
