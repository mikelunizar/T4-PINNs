import pytorch_lightning as pl
import torch
import numpy as np


class Solver(pl.LightningModule):
    """ Authors: PhD Mikel M Iparraguirre | PhD Lucas Tesan
                 mikel.martinez@unizar.es | ltesan@unizar.es
        University of Zaragoza, Applied Mechanics Department (AMB)
    """
    def __init__(self, base_model, criterion=None, X_bc=None, u_bc=None, X_ic=None, u_ic=None,
                 lr=5e-4, optimizer=torch.optim.Adam, T=2):
        super(Solver, self).__init__()
        self.model = base_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.X_bc = torch.tensor(X_bc, dtype=torch.float32)
        self.u_bc = torch.tensor(u_bc, dtype=torch.float32)
        self.X_ic = torch.tensor(X_ic, dtype=torch.float32)
        self.u_ic = torch.tensor(u_ic, dtype=torch.float32)

        self.T = T
        self.stored_predictions = dict()

    def forward(self, x):
        x_inp, t_inp = x[:, 0:1], x[:, 1:]
        x = self.model(x)
        x = x * t_inp * (1 - x_inp) * (1 + x_inp) - torch.sin(torch.pi*x_inp)
        return x

    def training_step(self, batch, batch_idx):
        X_cp, y = batch
        losses = self.criterion(X_cp, self.X_bc, self.u_bc, self.X_ic, self.u_ic, self)
        for name_loss, value_loss in losses.items():
            self.log(f'train_{name_loss}', value_loss.cpu().detach().item(), on_epoch=True, on_step=False)
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        X_cp, y = batch
        losses = self.criterion(X_cp, self.X_bc, self.u_bc, self.X_ic, self.u_ic, self)
        for name_loss, value_loss in losses.items():
            self.log(f'val_{name_loss}', value_loss.cpu().detach().item(), on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        #lr_scheduler = {
        #    'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5),
        #    'monitor': 'train_loss'}
        #return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        return {'optimizer': optimizer,}

    def store_prediction(self):
        time = np.linspace(0, self.T, 100)
        time_t_tensor = torch.tensor(time, dtype=torch.float32).reshape(-1, 1).to(self.device)
        predicted_solution = self.forward(time_t_tensor)
        self.stored_predictions[f'{self.current_epoch}'] = predicted_solution.cpu().detach().numpy()