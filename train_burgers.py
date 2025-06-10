import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from src.solver import Solver
from src.loss_burgers import loss_pinn
from src.dataloader import build_dataloader


if __name__ == '__main__':

    Nx, Nt = 256, 100
    Nc = 2500
    epochs = 2000

    # Simulate real world scenario by random sampling data
    x = np.linspace(-1, 1, Nx)
    t = np.linspace(0, 1, Nt)
    X, T = np.meshgrid(x, t)
    points = np.stack([X.ravel(), T.ravel()], axis=-1)  # shape (N, 2) where N = Nx * Nt

    # Define Boundary Conditions
    X_bc = points[np.argwhere((points[:, 0] == 1) | (points[:, 0] == -1)).reshape(-1)]  # bc u(1,t) = u(-1,t) = 0
    u_bc = np.zeros(X_bc.shape[0]).reshape(-1, 1)  # ic u(1,t) = u(-1, t) = 0
    # Define Initial  Conditions
    X_ic = points[np.argwhere(points[:, 1] == 0).reshape(-1)]  # ic u(x,0) = -sin(pi*x)
    u_ic = -np.sin(np.pi*X_ic[:, 0]).reshape(-1, 1)
    # Define Collocation Points where to evaluate PDE
    X_cp = points[np.random.choice(len(points), Nc, replace=False)]

    # Split the data train, val (80-20%)
    train_X_cp, valid_X_cp, train_u, valid_u = train_test_split(X_cp, np.zeros(len(X_cp)), test_size=0.2, random_state=52)

    # Plot collocation points and BC and IC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Left subplot - BC and IC
    sc1 = ax1.scatter(X_ic[:, 0:1], X_ic[:, 1:], c=u_ic, cmap='jet',
                      vmax=(max(u_ic.max(), u_bc.max())), vmin=min(u_ic.min(), u_bc.min()))
    ax1.scatter(X_bc[:, 0:1], X_bc[:, 1:], c=u_bc, cmap='jet',
            vmax=(max(u_ic.max(), u_bc.max())), vmin=min(u_ic.min(), u_bc.min()))
    ax1.set_title('Boundary and Initial Conditions')
    ax1.set_ylabel('Time (s)')
    ax1.set_xlabel('X (m)')
    fig.colorbar(sc1, ax=ax1)
    # Collocation Points
    ax2.scatter(train_X_cp[:, 0:1], train_X_cp[:, 1:], c='blue', marker='+', label='Train')
    ax2.scatter(valid_X_cp[:, 0:1], valid_X_cp[:, 1:], c='orange', marker='o', label='Valid')
    ax2.set_title('Collocation Points')
    ax2.set_ylabel('Time (s)')
    ax2.set_xlabel('X (m)')
    plt.legend()
    plt.show()

    # Build loaders
    train_loader = build_dataloader(train_X_cp, train_u, batch_size=128)
    valid_loader = build_dataloader(valid_X_cp, valid_u, batch_size=128)

    # Instantiate the model
    neural_network = torch.nn.Sequential(torch.nn.Linear(2, 64),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(64, 64),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(64, 64),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(64, 1))

    # Regularization techniques
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min')
    # Model setup
    model = Solver(neural_network, criterion=loss_pinn,
                   X_bc=X_bc, u_bc=u_bc, X_ic=X_ic, u_ic=u_ic,
                   lr=1e-3, optimizer=torch.optim.Adam)
    # Trainer setup
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         max_epochs=epochs, logger=WandbLogger(name='ViscousBurgers', project='T4-PINNs'),
                         callbacks=[early_stopping], check_val_every_n_epoch=25)
    # Training the model
    trainer.fit(model, train_loader, valid_loader)
    # Save the model
    torch.save(model.state_dict(), './model_pinn.pth')

    # INFERENCE HIGH RESOLUTION
    X_high_res = torch.tensor(points, dtype=torch.float32)
    prediction_high_res = model(X_high_res).detach().numpy()

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    # Left subplot - prediction
    sc1 = ax1.scatter(X_high_res[:, 0:1], X_high_res[:, 1:], c=prediction_high_res, cmap='jet')
    ax1.set_title('Prediction')
    fig.colorbar(sc1, ax=ax1)
    # Centre subplot - target
    target_high_res = np.load('./data/burgers_sol.npy', allow_pickle=True)[-1]
    sc2 = ax2.scatter(X_high_res[:, 0:1], X_high_res[:, 1:], c=target_high_res, cmap='jet')
    ax2.set_title('Target')
    fig.colorbar(sc2, ax=ax2)
    # Right subplot - error
    sc3 = ax3.scatter(X_high_res[:, 0:1], X_high_res[:, 1:], c=target_high_res - prediction_high_res,
                      vmax=0.01, vmin=-0.01, s=5, cmap='seismic')
    ax3.set_title('Error (Target - Prediction)')
    fig.colorbar(sc3, ax=ax3)
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

