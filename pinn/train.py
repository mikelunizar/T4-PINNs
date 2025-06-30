import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import numpy as np

from pinn.src.residuals import residual_loss
from pinn.src.pde_solver import PDESolver
from pinn.src.data_generator import collocation_points_generation, boundary_condition_points, initial_condition_points
from pinn.src.visuals import plot_collocation_setup




# Reproducibility
np.random.seed(53)
pl.seed_everything(53)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


if __name__ == '__main__':

    # PDE Problem Domain
    Nx, Nt = 300, 200
    # set grid PDE domain at resolution Nx, Nt
    x = np.linspace(-1, 1, Nx)
    t = np.linspace(0, 1, Nt)
    X, T = np.meshgrid(x, t)
    xt_grid = np.stack([X.ravel(), T.ravel()], axis=-1)  # shape (N, 2) where N = Nx * Nt

    # Generate collocation points
    Ncp = 500  # Number of Collocation points
    coll_points_train, coll_points_val = collocation_points_generation(xt_grid, Ncp, val_ratio=0.2)

    # Generate boundary condition points (x,t) and solution u(x,t)
    Nbc = 100  # Number of Boundary condition points
    xt_bc, u_bc = boundary_condition_points(Nbc, xt_grid)

    # Generate initial condition points (x,t) and solution u(x,t)
    Nic = 100  # Number Initial condition points
    xt_ic, u_ic = initial_condition_points(Nic, xt_grid)

    # Plot setup problem
    plot_collocation_setup(xt_grid, coll_points_train, coll_points_val, xt_bc, u_bc, xt_ic, u_ic)

    # Model Architecture
    # input, output and hidden size
    input_size = 2
    ouput_size = 1
    hiden_size = 64
    # NN model
    model = torch.nn.Sequential(torch.nn.Linear(input_size, hiden_size),
                                torch.nn.Tanh(),
                                torch.nn.Linear(hiden_size, hiden_size),
                                torch.nn.Tanh(),
                                torch.nn.Linear(hiden_size, hiden_size),
                                torch.nn.Tanh(),
                                torch.nn.Linear(hiden_size, hiden_size),
                                torch.nn.Tanh(),
                                torch.nn.Linear(hiden_size, ouput_size))

    # Set up Optimization process
    # hyperparameters
    opt = torch.optim.Adam
    lr = 1e-3
    bs = 128
    foldername = f'bs={bs}_lr={lr}_Ncp={Ncp}'  # Run name

    # Solver setup
    pde_solver = PDESolver(model, residual=residual_loss,
                           xt_bc=xt_bc, u_bc=u_bc, xt_ic=xt_ic, u_ic=u_ic,
                           lr=lr, optimizer=opt, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Callbacks
    early_stopping = pl.callbacks.EarlyStopping(monitor='valid_residual', patience=25, mode='min')  # regularization technique
    checkpoint_callback = ModelCheckpoint( monitor='valid_residual', dirpath=f'checkpoints/{foldername}', filename='best_model', save_top_k=1, mode='min', save_weights_only=True)
    logger = WandbLogger(name=foldername, project='T4-PINNs')  # W&B logger

    # Trainer setup
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         max_epochs=2000,
                         logger=logger, callbacks=[early_stopping],
                         check_val_every_n_epoch=10)

    # dataloaders
    train_loader = DataLoader(torch.tensor(coll_points_train, dtype=torch.float32), batch_size=32, shuffle=True)
    valid_loader = DataLoader(torch.tensor(coll_points_val, dtype=torch.float32), batch_size=32, shuffle=False)

    # Training the model
    trainer.fit(pde_solver, train_loader, valid_loader)

    # perform prediction
    u_grid_hat = model(torch.tensor(xt_grid, dtype=torch.float32)).detach()

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1) = plt.subplots(1, 1)
    # Left subplot - prediction
    sc1 = ax1.scatter(xt_grid[:, 0:1], xt_grid[:, 1:], c=u_grid_hat, cmap='jet')
    ax1.set_title(f'Resolution {Nx}x{Nt}')
    fig.colorbar(sc1, ax=ax1)
    plt.show()

