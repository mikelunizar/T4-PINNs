import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def collocation_points_generation(Nx, Nt, Nc, ratio):

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
    ax2.scatter(train_X_cp[:, 0:1], train_X_cp[:, 1:], c='blue', marker='*', label='Train', alpha=0.5)
    ax2.scatter(valid_X_cp[:, 0:1], valid_X_cp[:, 1:], c='orange', marker='o', label='Valid', alpha=0.5)
    ax2.set_title('Collocation Points')
    ax2.set_ylabel('Time (s)')
    ax2.set_xlabel('X (m)')
    plt.legend()
    plt.show()

    return (train_X_cp, train_u), (valid_X_cp, valid_u), points, (X_bc, u_bc), (X_ic, u_ic)