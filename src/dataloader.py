import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch


def build_dataloader(X_cp, u, batch_size=128):
    # Convert data to PyTorch tensors
    X_cp = torch.tensor(X_cp, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    # Create TensorDatasets
    dataset = TensorDataset(X_cp, u)
    # Create DataLoaders
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader



