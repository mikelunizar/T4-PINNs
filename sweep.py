import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils import ModelSaveTopK

from src.solver import Solver
from src.loss_burgers import loss_pinn
from src.dataloader import build_dataloader


np.random.seed(53)
pl.seed_everything(53)

if __name__ == '__main__':

    Nx, Nt = 256, 100
    Nc = 2500
    epochs = 5000
    bs = 128
    lr = 1e-2
    with_enc = True
    ratio = 0.5

    foldername = f'bs={bs}_lr={lr}_ratio={ratio}_withEnc={with_enc}'

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
    train_X_cp, train_u = train_X_cp[:int(len(train_X_cp)*ratio)], train_X_cp[:int(len(train_X_cp)*ratio)]

    # Build loaders
    train_loader = build_dataloader(train_X_cp, train_u, batch_size=bs)
    valid_loader = build_dataloader(valid_X_cp, valid_u, batch_size=bs)

    # Instantiate the model
    neural_network = torch.nn.Sequential(torch.nn.Linear(22 if with_enc else 2, 64),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(64, 64),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(64, 64),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(64, 1))
    # Best saved model
    save_topk = ModelSaveTopK(dirpath=str(f'outputs/{foldername}'), monitor='val_loss', mode='min', topk=1)
    # Model setup
    model = Solver(neural_network, criterion=loss_pinn,
                   X_bc=X_bc, u_bc=u_bc, X_ic=X_ic, u_ic=u_ic, points=points,
                   lr=lr, optimizer=torch.optim.Adam, with_enc=with_enc)
    # Trainer setup
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         max_epochs=epochs, logger=WandbLogger(name=foldername, project='T4-PINNs'),
                         callbacks=[save_topk], check_val_every_n_epoch=25)
    # Training the model
    trainer.fit(model, train_loader, valid_loader)
    # Load best model and Evaluate
    model.load_state_dict(torch.load(f'outputs/{foldername}/topk1.pth', weights_only=True))
    # evaluate model wiht target
    target_high_res = np.load('./data/burgers_sol.npy', allow_pickle=True)[-1]
    prediction_high_res = model(torch.tensor(points, dtype=torch.float32)).detach().numpy()
    trainer.log('test_rmse', np.sqrt(np.mean((target_high_res-prediction_high_res)**2)), on_epoch=True, on_step=False)







