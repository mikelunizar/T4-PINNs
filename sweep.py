import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from pinn.residuals import residual_loss
from pinn.pde_solver import PDESolver
from pinn.data_generator import collocation_points_generation, boundary_condition_points, initial_condition_points
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader


# PDE problem domain setup (same as before)
Nx, Nt = 300, 200
x = np.linspace(-1, 1, Nx)
t = np.linspace(0, 1, Nt)
X, T = np.meshgrid(x, t)
xt_grid = np.stack([X.ravel(), T.ravel()], axis=-1)

Ncp = 500
coll_points_train, coll_points_val = collocation_points_generation(xt_grid, Ncp, val_ratio=0.2)
Nbc = 100
xt_bc, u_bc = boundary_condition_points(Nbc, xt_grid)
Nic = 100
xt_ic, u_ic = initial_condition_points(Nic, xt_grid)


def train(config=None):

    wandb.init(project="T4-PINNs", config=config)
    config = wandb.config

    # Model architecture - using config.hidden_size
    model = torch.nn.Sequential(
        torch.nn.Linear(2, config.hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(config.hidden_size, config.hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(config.hidden_size, config.hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(config.hidden_size, 1),
    )

    # PDE solver setup
    pde_solver = PDESolver(model, residual=residual_loss,
                           xt_bc=xt_bc, u_bc=u_bc,
                           xt_ic=xt_ic, u_ic=u_ic,
                           lr=config.lr, optimizer=torch.optim.Adam,
                           device='cuda' if torch.cuda.is_available() else 'cpu')

    # Callbacks
    early_stopping = EarlyStopping(monitor='valid_residual', patience=config.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_residual',
        dirpath=f'checkpoints/bs={config.batch_size}_lr={config.lr}_hs={config.hidden_size}',
        filename='best_model',
        save_top_k=1,
        mode='min',
        save_weights_only=True)

    logger = WandbLogger(project="T4-PINNs", name=f"bs={config.batch_size}_lr={config.lr}_hs={config.hidden_size}")

    # Trainer
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         max_epochs=config.max_epochs,
                         logger=logger,
                         callbacks=[early_stopping, checkpoint_callback],
                         check_val_every_n_epoch=10)

    # DataLoaders with batch size from config
    train_loader = DataLoader(torch.tensor(coll_points_train, dtype=torch.float32), batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(torch.tensor(coll_points_val, dtype=torch.float32), batch_size=config.batch_size, shuffle=False)

    trainer.fit(pde_solver, train_loader, valid_loader)
    wandb.finish()


if __name__ == "__main__":
    train()
