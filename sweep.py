import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from src.utils import ModelSaveTopK

from src.solver import Solver
from src.loss_burgers import loss_pinn
from src.dataloader import build_dataloader
from src.data_generator import collocation_points_generation
import wandb

np.random.seed(53)
pl.seed_everything(53)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(config=None):
    # Override default hyperparameters with sweep config if provided
    with wandb.init(config=config):
        args = wandb.config

    Nx, Nt = 256, 100
    Nc = 2500
    epochs = 3000
    bs = args.bs
    lr = args.lr
    with_enc = args.with_enc
    ratio = args.ratio
    opt = torch.optim.Adam if args.optimizer == 'adam' else torch.optim.SGD
    #opt = torch.optim.SGD

    # Run name
    foldername = f'bs={bs}_lr={lr}_ratio={ratio}_withEnc={with_enc}_opt={"adam"}'

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

    # Best saved model
    save_topk = ModelSaveTopK(dirpath=str(f'outputs/{foldername}'), monitor='val_loss', mode='min', topk=1)
    # Model setup
    model = Solver(neural_network, criterion=loss_pinn,
                   X_bc=bc[0], u_bc=bc[1], X_ic=ic[0], u_ic=ic[1], points=points,
                   lr=lr, optimizer=opt, with_enc=with_enc, device=device)
    # Trainer setup
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         max_epochs=epochs, logger=WandbLogger(name=foldername, project='T4-PINNs'),
                         callbacks=[save_topk], check_val_every_n_epoch=25)
    # Training the model
    trainer.fit(model, train_loader, valid_loader)

    # Load best model and Evaluate
    model.load_state_dict(torch.load(f'outputs/{foldername}/topk1.pth', weights_only=True))
    # evaluate model with target
    target_high_res = np.load('./data/burgers_sol.npy', allow_pickle=True)[-1]
    test_loader = build_dataloader(points, target_high_res, batch_size=len(target_high_res), shuffle=False)
    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    train()

