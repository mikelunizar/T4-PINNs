import torch
import numpy as np

mu, mass, k = 15, 5, 400  # μ (mu),  m (mass), k (oscillator constant)


def loss_osc_ode(t, solver):
    # Ensure t requires gradient computation
    t.requires_grad_(True)
    # Enable gradient computation manually
    with torch.set_grad_enabled(True):
        # Compute the neural network output u(t)
        x = solver.forward(t.reshape(-1, 1))
        # Compute the derivative of u with respect to t
        dx_dt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        d2x_dt2 = torch.autograd.grad(dx_dt, t, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
        # Compute the ODE loss: difference between u_t and the target function's derivative
        ode_loss = mass * d2x_dt2 + mu * dx_dt + k*x
        ode_loss = torch.mean(ode_loss ** 2)

    return ode_loss


def loss_osc_ic_pos(solver):
    t_0 = torch.zeros((1, 1), dtype=torch.float32).to(solver.device)
    one = torch.ones((1, 1), dtype=torch.float32).to(solver.device)
    ic_pos_loss = solver.forward(t_0) - one
    ic_pos_loss = torch.mean(ic_pos_loss ** 2)
    return ic_pos_loss


def loss_osc_ic_vel(solver):
    # Define initial condition (t=0) and the target value at t=0
    t_0 = torch.zeros((1, 1), dtype=torch.float32).to(solver.device)
    t_0.requires_grad_(True)
    # Enable gradient computation manually
    with torch.set_grad_enabled(True):
        # Compute the neural network output u(t)
        x = solver.forward(t_0)
        # Compute the derivative of u with respect to t
        dx_dt = torch.autograd.grad(x, t_0, grad_outputs=torch.ones_like(x), create_graph=True)[0]

    ic_vel_loss = torch.mean(dx_dt**2)

    return ic_vel_loss


def loss_pinn(t, y, solver):

    # ode-supervised loss
    physic_t = np.linspace(0, solver.T, 25).reshape(-1, 1)
    physic_t = torch.tensor(physic_t, dtype=torch.float32).to(solver.device)
    #physic_t = t
    # compute losses
    ode_osc_loss = loss_osc_ode(physic_t, solver)
    ic_pos_osc_loss = loss_osc_ic_pos(solver)
    ic_vel_osc_loss = loss_osc_ic_vel(solver)
    # sum weighted losses
    delta = 1e-7
    pinn_loss = delta*ode_osc_loss + ic_pos_osc_loss + ic_vel_osc_loss

    return {'loss': pinn_loss, 'ode_loss': ode_osc_loss, 'ic_pos_loss': ic_pos_osc_loss, 'ic_vel_loss': ic_vel_osc_loss}
