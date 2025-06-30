import torch
import pytorch_lightning as pl


class PDESolver(pl.LightningModule):
    """
    Authors: Mikel M Iparraguirre | Lucas Tesan
             mikel.martinez@unizar.es | ltesan@unizar.es
    University of Zaragoza, Applied Mechanics Department (AMB)

    Physics-Informed Neural Network solver module implemented with PyTorch Lightning.
    """

    def __init__(self, model, residual=None, xt_bc=None, u_bc=None, xt_ic=None, u_ic=None, points=None,
                 lr=1e-3, optimizer=torch.optim.Adam, with_enc=False, device='cpu'):
        """
        Initialize the PDE solver module.

        Parameters:
        - model: Neural network model approximating the solution u(x,t).
        - residual: Function to compute residual losses for PDE, BC, IC.
        - xt_bc, u_bc: Boundary condition points and corresponding values.
        - xt_ic, u_ic: Initial condition points and corresponding values.
        - lr: Learning rate for optimizer.
        - optimizer: Optimizer class (default: Adam).
        - with_enc: Whether to apply frequency encoding to inputs.
        - device: Device to load tensors on ('cpu' or 'cuda').
        """
        super().__init__()

        self.model = model  # Neural network model
        self.residual = residual  # Residual loss function
        self.optimizer = optimizer  # Optimizer class
        self.lr = lr  # Learning rate
        self.with_enc = with_enc  # Use Fourier feature encoding flag

        # Store BC and IC points and values as tensors on the target device
        self.xt_bc = torch.tensor(xt_bc, dtype=torch.float32).to(device)
        self.u_bc = torch.tensor(u_bc, dtype=torch.float32).to(device)
        self.xt_ic = torch.tensor(xt_ic, dtype=torch.float32).to(device)
        self.u_ic = torch.tensor(u_ic, dtype=torch.float32).to(device)

    def forward(self, x):
        """
        Forward pass through the network.
        Optionally applies Fourier feature encoding to inputs.
        """
        if self.with_enc:
            x = self.__encode_to_frequency_domain(x)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Executes one training step.
        Computes residuals on collocation points batch, logs losses.
        """
        xt_cp = batch  # collocation points batch

        # Compute residual losses (PDE, BC, IC)
        residual_total = self.residual(xt_cp, self.xt_bc, self.u_bc, self.xt_ic, self.u_ic, self)

        # Log each residual component
        for name_residual, value_residual in residual_total.items():
            self.log(f'train_{name_residual}', value_residual.cpu().detach().item(), on_epoch=True, on_step=False)

        # Return total residual loss for optimization
        return residual_total['residual']

    def validation_step(self, batch, batch_idx):
        """
        Executes one validation step.
        Similar to training_step but logs validation losses.
        """
        xt_cp = batch

        residual_total = self.residual(xt_cp, self.xt_bc, self.u_bc, self.xt_ic, self.u_ic, self)

        for name_residual, value_residual in residual_total.items():
            self.log(f'valid_{name_residual}', value_residual.cpu().detach().item(), on_epoch=True, on_step=False)

        return residual_total['residual']

    def configure_optimizers(self):
        """
        Setup optimizer and learning rate scheduler.
        Here, using Cosine Annealing LR scheduler over the training epochs.
        """
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=1e-5),
            'interval': 'epoch',
            'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def __encode_to_frequency_domain(self, x, num_freq=5):
        """
        Applies Fourier feature encoding to input coordinates.

        Args:
        - x: Tensor of shape (N, 2), where columns are (x, t)
        - num_freq: Number of frequency bands to encode

        Returns:
        - Encoded tensor with sinusoidal features concatenated.
        """
        # Generate frequency bands: 2^0, 2^1, ..., 2^(num_freq-1)
        ws = 2.0 ** torch.arange(0, num_freq, dtype=torch.float32, device=x.device).view(1, -1)

        # Separate spatial and temporal coordinates
        x_x, x_t = x[:, 0:1], x[:, 1:]

        # Compute sin and cos features for spatial coordinate
        sin_x = torch.sin(x_x * ws)
        cos_x = torch.cos(x_x * ws)
        # Compute sin and cos features for temporal coordinate
        sin_t = torch.sin(x_t * ws)
        cos_t = torch.cos(x_t * ws)

        # Concatenate original inputs with Fourier features
        encoded = torch.cat([x, sin_x, cos_x, sin_t, cos_t], dim=-1)

        return encoded