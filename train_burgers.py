import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import matplotlib.pyplot as plt

from src.solver import Solver
from src.loss_burgers import loss_pinn
from src.dataloader import build_dataloader
from src.visuals import create_prediction_gif
from src.utils import ModelSaveTopK
from src.data_generator import collocation_points_generation


np.random.seed(53)
pl.seed_everything(53)

if __name__ == '__main__':

    Nx, Nt = 256, 100
    Nc = 2500
    epochs = 200
    ratio = 0.1
    bs = 128
    lr = 1e-2
    with_enc = False
    opt = torch.optim.SGD

    # Run name
    foldername = f'bs={bs}_lr={lr}_ratio={ratio}_withEnc={with_enc}'

    # Data generation
    train_data, valid_data, points, ic, bc = collocation_points_generation(Nx, Nt, Nc, ratio)
    # Build loaders
    train_loader = build_dataloader(train_data[0], train_data[1], batch_size=bs)
    valid_loader = build_dataloader(valid_data[0], valid_data[1], batch_size=bs)
    # Instantiate the model
    neural_network = torch.nn.Sequential(torch.nn.Linear(22 if with_enc else 2, 50),
                                         torch.nn.Tanh(),
                                         torch.nn.Linear(50, 50),
                                         torch.nn.Tanh(),
                                         torch.nn.Linear(50, 50),
                                         torch.nn.Tanh(),
                                         torch.nn.Linear(50, 1))

    # Regularization techniques
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    # Best saved model
    save_topk = ModelSaveTopK(dirpath=str(f'outputs/{foldername}'), monitor='val_loss', mode='min', topk=1)
    # Model setup
    model = Solver(neural_network, criterion=loss_pinn,
                   X_bc=bc[0], u_bc=bc[1], X_ic=ic[0], u_ic=ic[1], points=points,
                   lr=lr, optimizer=opt, with_enc=with_enc)
    # Trainer setup
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         max_epochs=epochs, logger=WandbLogger(name=foldername, project='T4-PINNs'),
                         callbacks=[early_stopping, save_topk], check_val_every_n_epoch=25)
    # Training the model
    trainer.fit(model, train_loader, valid_loader)

    # Load best model and Evaluate
    model.load_state_dict(torch.load(f'outputs/{foldername}/topk1.pth', weights_only=True))

    # evaluate model with target
    target_high_res = np.load('./data/burgers_sol.npy', allow_pickle=True)[-1]
    test_loader = build_dataloader(points, target_high_res, batch_size=len(target_high_res), shuffle=False)
    trainer.test(model, dataloaders=test_loader)
    #create_prediction_gif(model.stored_predictions, points, x_cp=train_data[0], output_path=f'outputs/{foldername}/evolution.gif')

    # INFERENCE SUPER - HIGH RESOLUTION
    Nx, Nt = 1000, 1000
    x = np.linspace(-1, 1, Nx)
    t = np.linspace(0, 1, Nt)
    X, T = np.meshgrid(x, t)
    points_super_high_res = np.stack([X.ravel(), T.ravel()], axis=-1)

    X_super_high_res = torch.tensor(points_super_high_res, dtype=torch.float32)
    prediction_super_high_res = model(X_super_high_res).detach().numpy()
    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1) = plt.subplots(1, 1, figsize=(12, 5))
    # Left subplot - prediction
    sc1 = ax1.scatter(X_super_high_res[:, 0:1], X_super_high_res[:, 1:], c=prediction_super_high_res, cmap='jet')
    ax1.set_title(f'Super High Resolution {Nx}x{Nt}')
    fig.colorbar(sc1, ax=ax1)
    plt.show()

    # INFERENCE HIGH RESOLUTION
    X_high_res = torch.tensor(points, dtype=torch.float32)
    model.eval()
    prediction_high_res = model(X_high_res).detach().numpy()

    # Create a figure with 1 row and 2 columns of subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    # Left subplot - prediction
    sc1 = ax1.scatter(X_high_res[:, 0:1], X_high_res[:, 1:], c=prediction_high_res, cmap='plasma')
    ax1.set_title('Prediction')
    fig.colorbar(sc1, ax=ax1)
    # Centre subplot - target
    target_high_res = np.load('./data/burgers_sol.npy', allow_pickle=True)[-1]
    sc2 = ax2.scatter(X_high_res[:, 0:1], X_high_res[:, 1:], c=target_high_res, cmap='plasma')
    ax2.set_title('Target')
    fig.colorbar(sc2, ax=ax2)
    # Right subplot - error
    sc3 = ax3.scatter(X_high_res[:, 0:1], X_high_res[:, 1:], c=(target_high_res - prediction_high_res),
                      vmax=0.01, vmin=-0.01, cmap='seismic')
    ax3.set_title('Error (Target - Prediction)')
    fig.colorbar(sc3, ax=ax3)
    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

