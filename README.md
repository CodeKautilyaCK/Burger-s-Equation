# Burgers’ Equation: FDM vs PINN

This project demonstrates the 1-D **Burgers’ equation** solved using:

1. **Finite-Difference Method (FDM)** – classical numerical solver.  
2. **Physics-Informed Neural Network (PINN)** – AI-driven solution that learns the physics directly.

---

## 📚 Problem Description

The Burgers’ equation models **velocity evolution under nonlinear convection and diffusion**. It is often used as a benchmark for CFD solvers and to test emerging AI-based PDE solvers.

---

## 🛠️ Features

- Solve Burgers’ equation using FDM on a 1-D spatial grid.  
- Train a PINN to approximate the solution directly from the PDE.  
- Generate **animated comparison** between FDM and PINN solutions.  
- Lightweight Python implementation using **PyTorch** and **Matplotlib**.  

