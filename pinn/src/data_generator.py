import numpy as np
from sklearn.model_selection import train_test_split


def collocation_points_generation(xt_grid, Ncp, val_ratio=0.2):
    # Sample Ncp from grid where to evaluate PDE
    xt_cp = xt_grid[np.random.choice(len(xt_grid), Ncp, replace=False)]
    # Split the data train, val (80-20%)
    train_xt_cp, valid_xt_cp, _, _ = train_test_split(xt_cp, np.zeros(len(xt_cp)), test_size=val_ratio, random_state=52)

    return train_xt_cp, valid_xt_cp


def boundary_condition_points(Nbc, xt_grid):
    # Define Boundary Conditions
    xt_bc = xt_grid[np.argwhere((xt_grid[:, 0] == 1) | (xt_grid[:, 0] == -1)).reshape(-1)]  # all points at x=-1 and x=1
    xt_bc = xt_bc[np.random.choice(len(xt_bc), Nbc, replace=False)]
    u_bc = np.zeros(xt_bc.shape[0]).reshape(-1, 1)  # set bc of u(-1, t) = u(1, t) = 0

    return xt_bc, u_bc


def initial_condition_points(Nic, xt_grid):
    # Define Initial  Conditions
    xt_ic = xt_grid[np.argwhere(xt_grid[:, 1] == 0).reshape(-1)]  # All points with t=0
    xt_ic = xt_ic[np.random.choice(len(xt_ic), Nic, replace=False)]
    u_ic = -np.sin(np.pi*xt_ic[:, 0]).reshape(-1, 1)  # set ic u(x, 0) = - sin(pi*x)

    return xt_ic, u_ic