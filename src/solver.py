import numpy as np
import pytorch_lightning as pl
import torch


class Solver(pl.LightningModule):
    """ Authors: PhD Mikel M Iparraguirre | PhD Lucas Tesan
                 mikel.martinez@unizar.es | ltesan@unizar.es
        University of Zaragoza, Applied Mechanics Department (AMB)
    """
    def __init__(self, base_model, criterion=None, X_bc=None, u_bc=None, X_ic=None, u_ic=None, points=None,
                 lr=1e-3, optimizer=torch.optim.Adam, with_enc=False, device='cpu'):
        super(Solver, self).__init__()

        self.model = base_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.with_enc = with_enc

        self.X_bc = torch.tensor(X_bc, dtype=torch.float32).to(device)
        self.u_bc = torch.tensor(u_bc, dtype=torch.float32).to(device)
        self.X_ic = torch.tensor(X_ic, dtype=torch.float32).to(device)
        self.u_ic = torch.tensor(u_ic, dtype=torch.float32).to(device)

        self.points = torch.tensor(points, dtype=torch.float32).to(device) if points is not None else points
        self.stored_predictions = dict()

    def forward(self, x):
        #x_inp, t_inp = x[:, 0:1], x[:, 1:]
        if self.with_enc:
            x = self.encode_to_frequency_domain(x)
        x = self.model(x)
        #x = x * t_inp * (1 - x_inp) * (1 + x_inp) - torch.sin(torch.pi*x_inp)
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
        if self.points is not None:
            self.store_prediction()

    def test_step(self, batch, batch_id):
        x_cp, target = batch[0], batch[1]
        prediction = self.forward(x_cp.clone().detach())
        rmse = torch.sqrt(torch.mean((prediction-target)**2)).detach().item()
        self.test_error = rmse

    def on_test_epoch_end(self):
        print(f'RMSE error HQ solution: {self.test_error}')
        self.log('test error HQ', self.test_error)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr/100),
            'monitor': 'train_loss'}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        #return {'optimizer': optimizer,}

    def encode_to_frequency_domain(self, x, num_freq=5):

        # Create frequency bands (2^0 to 2^(num_freq-1))
        ws = 2.0 ** torch.arange(0, num_freq, dtype=torch.float32, device=x.device).view(1, -1)

        # Split spatial (x) and temporal (t) components
        x_x, x_t = x[:, 0:1], x[:, 1:]

        # Compute Fourier features
        sin_x = torch.sin(x_x * ws)  # shape (N, num_freq)
        cos_x = torch.cos(x_x * ws)
        sin_t = torch.sin(x_t * ws)
        cos_t = torch.cos(x_t * ws)

        # Concatenate all features
        encoded = torch.cat([x, sin_x, cos_x, sin_t, cos_t], dim=-1)

        return encoded

    def store_prediction(self):

        predicted_solution = self.forward(self.points)
        self.stored_predictions[f'{self.current_epoch}'] = predicted_solution.cpu().detach().numpy()

        # fig, (ax1) = plt.subplots(1, 1, figsize=(12, 5))
        # # Left subplot - prediction
        # sc1 = ax1.scatter(self.points[:, 0:1].cpu().detach().numpy(), self.points[:, 1:].cpu().detach().numpy(), c=predicted_solution, cmap='plasma', label='u(x,t)')
        # ax1.set_title('Prediction')
        # ax1.set_ylabel('Time(s)')
        # ax1.set_xlabel('X(m)')
        # plt.legend()
        # fig.colorbar(sc1, ax=ax1)
        # # Log to wandb
        # if wandb.run is not None:  # Check if wandb is initialized
        #     wandb.log({
        #         "Prediction": wandb.Image(fig),
        #         "epoch": self.current_epoch
        #     })
        #
        # plt.close(fig)  # Close the figure to free memory

