# MeZO: Memory-Efficient Zeroth-Order Optimization

This directory contains the implementation of the MeZO algorithm.

## Description

MeZO is a memory-efficient zeroth-order optimizer designed for fine-tuning large-scale models where traditional backpropagation is memory-intensive. It is a variant of Simultaneous Perturbation Stochastic Approximation (SPSA) that avoids storing the perturbation vector, instead regenerating it from a seed. This results in $O(1)$ memory complexity for the optimizer's state.

The key idea is to estimate the projected gradient using two forward passes and then update the parameters in-place, all while re-computing the Gaussian perturbation vector on-the-fly.

## Key Features

- **Memory Efficiency:** Requires only $O(1)$ auxiliary memory, making it suitable for extremely large models.
- **In-place Updates:** All parameter updates are performed in-place to minimize memory overhead.
- **Zeroth-Order:** Does not require explicit gradient calculations, making it applicable to non-differentiable objectives.

## Reference

- Malladi, S., Chen, T., Bubeck, S., & Lee, Y. T. (2023). *MeZO: Memory-Efficient Zeroth-Order Optimization for Fine-tuning Large Language Models*. arXiv preprint arXiv:2305.17333.
