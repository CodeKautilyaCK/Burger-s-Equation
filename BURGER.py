"""
main.py
Physics-Informed Neural Network vs Finite-Difference
Burgers' equation (1-D, viscous)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def burgers_fd(nx=256, nt=1000, nu=0.01/np.pi, dt=None):
    x = np.linspace(0, 1, nx)
    dx = x[1] - x[0]
    u = -np.sin(np.pi * x)
    if dt is None:
        dt = 0.4 * dx**2 / nu       
    snapshots = [u.copy()]
    t_store = [0.0]

    for n in range(nt):
        u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        u_x  = (u - np.roll(u, 1)) / dx
        u    = u + dt * (nu * u_xx - u * u_x)
        u[0] = u[-1] = 0
        if n % 10 == 0:
            snapshots.append(u.copy())
            t_store.append((n + 1) * dt)
    return x, np.array(snapshots), np.array(t_store)


class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.net.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x):
        for l in self.net[:-1]:
            x = torch.tanh(l(x))
        return self.net[-1](x)

def pde_residual(model, x, t, nu):
    xt = torch.cat([x, t], dim=1).requires_grad_(True)
    u = model(xt)
    grads = grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return u_t + u * u_x - nu * u_xx

def train_pinn(nu=0.01/np.pi, epochs=20000, device="cpu"):
    model = PINN([2, 64, 64, 64, 64, 1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

   
    N_f = 20000
    x_f = torch.rand(N_f, 1, device=device)
    t_f = torch.rand(N_f, 1, device=device)

    x_i = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
    u_i = -torch.sin(np.pi * x_i)
    t_b = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
    zeros = torch.zeros_like(t_b)

    for epoch in range(epochs):
        opt.zero_grad()
        u_pred_i = model(torch.cat([x_i, torch.zeros_like(x_i)], dim=1))
        loss_ic = torch.mean((u_pred_i - u_i)**2)

        u_left  = model(torch.cat([zeros, t_b], dim=1))
        u_right = model(torch.cat([zeros + 1.0, t_b], dim=1))
        loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)

        f = pde_residual(model, x_f, t_f, nu)
        loss_pde = torch.mean(f**2)

        loss = loss_ic + loss_bc + loss_pde
        loss.backward()
        opt.step()

        if epoch % 2000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.5e}")

    return model


def animate_comparison(model, x_fd, u_fd, t_fd, filename="burgers_comparison.mp4"):
    x = torch.tensor(x_fd, dtype=torch.float32).unsqueeze(1)
    times = torch.tensor(t_fd, dtype=torch.float32).unsqueeze(1)
    X, T = np.meshgrid(x_fd, t_fd)
    xt = torch.tensor(np.c_[X.ravel(), T.ravel()], dtype=torch.float32)
    with torch.no_grad():
        u_pinn = model(xt).cpu().numpy().reshape(len(t_fd), len(x_fd))

    fig, ax = plt.subplots()
    line_fd,   = ax.plot([], [], 'b-', label='FDM')
    line_pinn, = ax.plot([], [], 'r--', label='PINN')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.legend()

    def update(frame):
        line_fd.set_data(x_fd, u_fd[frame])
        line_pinn.set_data(x_fd, u_pinn[frame])
        ax.set_title(f"t = {t_fd[frame]:.3f}")
        return line_fd, line_pinn

    ani = FuncAnimation(fig, update, frames=len(t_fd), blit=True)
    ani.save(filename, fps=15)
    plt.close(fig)
    print(f"Animation saved to {filename}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running finite-difference solver...")
    x_fd, u_fd, t_fd = burgers_fd()

    print("Training PINN (this may take a while)...")
    model = train_pinn(device=device)

    print("Creating animation...")
    animate_comparison(model, x_fd, u_fd, t_fd)
