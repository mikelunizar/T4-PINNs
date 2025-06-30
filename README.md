# 🧠 Physics-Informed Neural Networks (PINNs) for Solving 1D Burgers' Equation

This repository implements a **Physics-Informed Neural Network (PINN)** using **PyTorch Lightning** to solve the 1D **Burgers' equation**, a fundamental nonlinear PDE widely used in fluid mechanics and nonlinear wave phenomena.

The implementation is based on the seminal work of **Raissi, Perdikaris, and Karniadakis**:

> Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707.

---

## 📘 Problem Description

We solve the **viscous Burgers' equation** in one spatial dimension:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0
$$

- **Domain**: $x \in [-1, 1]$, $t \in [0, 1]$
- **Initial condition**: $u(x, 0) = -\sin(\pi x)$
- **Boundary conditions**: $u(-1, t) = u(1, t) = 0$
- $\nu$: Viscosity coefficient (default: 0.01/π)

---

## 📓 Interactive Tutorial Notebook

The repository includes a **Jupyter notebook tutorial** that walks you step-by-step through the design of a PINN-based solver for PDEs.

You'll learn:
- How to define **collocation points** in the domain
- How to build and evaluate the **residual function** using automatic differentiation
- How to incorporate **initial and boundary conditions** into the training loop
- How to visualize residuals, predictions, and training dynamics

The notebook features rich visualizations and is ideal for both beginners and researchers exploring PINNs.

---

## ⚡ Framework: PyTorch Lightning

This implementation is modular, clean, and fully based on **[PyTorch Lightning](https://www.pytorchlightning.ai/)** for:
- Training loop abstraction
- Early stopping and checkpointing
- GPU acceleration
- Integration with W&B for experiment tracking

---

## 🧮 Method

A fully connected neural network is trained to minimize the PDE residual using **automatic differentiation** via PyTorch. Training includes:
- Minimization of the PDE residual at randomly sampled collocation points
- Supervision with initial and boundary conditions
- Logging and visualization using **Weights & Biases (W&B)**

---

## 🏗️ Network Architecture
- **Input**: $(x, t) \in \mathbb{R}^2$
- **Hidden Layers**: 4 layers, each with 64 neurons
- **Activation**: `Tanh`
- **Output**: Scalar field $u(x, t)$

---

## 🚀 Installation

```bash
pip install git+https://github.com/mikelunizar/T4-PINNs.git