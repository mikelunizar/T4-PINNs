import torch

mu, mass, k = 15, 5, 400  # μ (mu),  m (mass), k (oscillator constant)
nu = 0.025


def loss_pde(X_cp, model):
    # Ensure t requires gradient computation
    x = X_cp[:, 0:1]
    t = X_cp[:, 1:]
    x.requires_grad_(True)
    t.requires_grad_(True)
    # Enable gradient computation manually
    with torch.set_grad_enabled(True):
        x_cp_grad = torch.cat([x, t], dim=-1)
        # Compute the neural network output u(t)
        u_hat = model.forward(x_cp_grad)
        # Compute the derivative of u with respect to t
        du_dt = torch.autograd.grad(u_hat, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
        dx_dx = torch.autograd.grad(u_hat, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        d2x_dx2 = torch.autograd.grad(dx_dx, x, grad_outputs=torch.ones_like(dx_dx), create_graph=True)[0]
        # Compute the ODE loss: difference between u_t and the target function's derivative
        loss_pde_ = torch.mean((du_dt + u_hat*dx_dx - nu*d2x_dx2)**2)

    return loss_pde_


def loss_ic(X_ic, u_ic, model):
    u_ic_hat = model.forward(X_ic)
    loss_ic_ = torch.mean((u_ic - u_ic_hat)**2)
    return loss_ic_


def loss_bc(X_bc, u_bc, model):
    u_bc_hat = model.forward(X_bc)
    loss_ic_ = torch.mean((u_bc_hat - u_bc)**2)
    return loss_ic_


def loss_pinn(X_cp, X_bc, u_bc, X_ic, u_ic, model):

    loss_pde_ = loss_pde(X_cp, model)
    loss_bc_ = loss_bc(X_bc, u_bc, model)
    loss_ic_ = loss_ic(X_ic, u_ic, model)

    loss = loss_pde_ + loss_bc_ + loss_ic_

    return {'loss': loss, 'loss_pde': loss_pde_, 'loss_bc': loss_bc_, 'loss_ic': loss_ic_}
