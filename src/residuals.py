
import torch
import numpy as np

# Physical parameters
nu = 0.025


def residual_pde(xt_cp, model):
    """
    Compute PDE residual loss at collocation points:
    """
    x = xt_cp[:, 0:1].clone().detach().requires_grad_(True)  # activate gradients
    t = xt_cp[:, 1:].clone().detach().requires_grad_(True)  # activate gradients

    with torch.enable_grad():
        inputs = torch.cat([x, t], dim=1)
        u_hat = model(inputs)

        # First derivatives
        du_dt = torch.autograd.grad(u_hat, t, grad_outputs=torch.ones_like(u_hat), create_graph=True)[0]
        du_dx = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(u_hat), create_graph=True)[0]

        # Second derivative wrt x
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

        # PDE residual: u_t + u * u_x - nu * u_xx = 0
        residual = du_dt + u_hat * du_dx - nu * d2u_dx2

        residual_pde_ = torch.mean(residual ** 2)

    return residual_pde_


def residual_ic(xt_ic, u_ic, model):
    """
    Compute MSE loss for Initial Condition:
    """
    u_pred = model(xt_ic)
    return torch.mean((u_ic - u_pred) ** 2)


def residual_bc(xt_bc, u_bc, model):
    """
    Compute MSE loss for Boundary Condition:
    """
    u_pred = model(xt_bc)
    return torch.mean((u_pred - u_bc) ** 2)


lambda_ic, lambda_bc = 1, 1  # set weights


def residual_loss(xt_cp, xt_bc, u_bc, xt_ic, u_ic, model):
    # PDE residual at collocation points
    residual_pde_cost = residual_pde(xt_cp, model)
    # Initial Conditions residual at i.c. points
    residual_ic_cost = residual_ic(xt_ic, u_ic, model)
    # Boundary Condition residual at b.c. points
    residual_bc_cost = residual_bc(xt_bc, u_bc, model)

    # Total residual: Weighted sum
    residual = residual_pde_cost + lambda_ic * residual_ic_cost +  lambda_bc * residual_bc_cost

    return {'residual': residual, 'residual_pde': residual_pde_cost, 'residual_bc': residual_bc_cost, 'residual_ic': residual_ic_cost}


def evaluate_residual_on_collocation_points(xt_grid, model, residual_pde_fn, batch_size=1024):
    """
    Evaluate residual_pde at each point in xt_grid using batching and original residual_pde function.

    Parameters:
        xt_grid (ndarray or tensor): shape (N, 2), with each row [x, t]
        model: the trained PINN model
        residual_pde_fn: original residual_pde function
        batch_size: number of points to evaluate per loop step

    Returns:
        residual_values: shape (N,) array of residuals
    """
    residual_values = []

    # Convert if necessary
    if isinstance(xt_grid, np.ndarray):
        xt_grid = torch.tensor(xt_grid, dtype=torch.float32)

    for i in range(0, len(xt_grid), batch_size):
        xt_batch = xt_grid[i:i + batch_size]

        batch_residuals = []
        for xt in xt_batch:
            r = residual_pde_fn(xt.unsqueeze(0), model)  # evaluate single point
            batch_residuals.append(r.item())

        residual_values.extend(batch_residuals)

    return np.array(residual_values)