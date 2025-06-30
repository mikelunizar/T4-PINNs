import matplotlib.pyplot as plt
import imageio
import os
import numpy as np


def create_prediction_gif(stored_predictions, points, x_cp=None, output_path='prediction_evolution.gif', fps=2):
    """
    Create a GIF showing the evolution of predictions over training epochs.

    Args:
        stored_predictions (dict): Dictionary with epoch numbers as keys and prediction arrays as values
        points (np.array): The coordinate points used for the predictions
        output_path (str): Path to save the GIF
        fps (int): Frames per second for the GIF
    """
    # Sort epochs in numerical order
    epochs = sorted([int(e) for e in stored_predictions.keys()])

    # Create a temporary directory for frames
    os.makedirs('temp_frames', exist_ok=True)

    # Generate each frame
    filenames = []
    for i, epoch in enumerate(epochs):
        # Create figure
        # Create a figure with 1 row and 2 columns of subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
        fig.suptitle(f'Model Predictions vs Target (Epoch {epoch})', fontsize=14, y=1.02)
        # Left subplot - prediction
        sc1 = ax1.scatter(points[:, 0:1], points[:, 1:], c=stored_predictions[str(epoch)], cmap='jet')
        ax1.set_title('Prediction')
        fig.colorbar(sc1, ax=ax1)
        # Centre subplot - target
        target_high_res = np.load('./data/burgers_sol.npy', allow_pickle=True)[-1]
        sc2 = ax2.scatter(points[:, 0:1], points[:, 1:], c=target_high_res, cmap='jet')
        ax2.set_title('Target')
        fig.colorbar(sc2, ax=ax2)
        # Right subplot - error
        sc3 = ax3.scatter(points[:, 0:1], points[:, 1:], c=(target_high_res - stored_predictions[str(epoch)]),
                          vmax=0.05, vmin=-0.05, cmap='seismic', alpha=0.65)
        if x_cp is not None:
            ax3.scatter(x_cp[:, 0:1], x_cp[:, 1:], c='black', marker='+', s=10, label='Col. Points')
        ax3.set_title('Error (Target - Prediction)')
        fig.colorbar(sc3, ax=ax3)
        # Adjust spacing between subplots
        plt.tight_layout()

        # Save frame
        filename = f'temp_frames/frame_{i:04d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        filenames.append(filename)
        plt.close(fig)

    # Create GIF from frames
    with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up temporary frames
    for filename in filenames:
        os.remove(filename)
    os.rmdir('temp_frames')

    print(f"GIF saved to {output_path}")


def plot_collocation_setup(xt_grid, coll_points_train, coll_points_val, xt_bc, u_bc, xt_ic, u_ic):
    plt.figure(figsize=(10, 6))

    plt.scatter(xt_grid[:, 0], xt_grid[:, 1], color='black', alpha=0.15, label='grid', s=1)

    # Plot collocation points
    plt.scatter(coll_points_train[:, 0], coll_points_train[:, 1], color='dodgerblue', s=10, label='Collocation (train)', alpha=0.6)
    plt.scatter(coll_points_val[:, 0], coll_points_val[:, 1], color='orange', s=10, label='Collocation (val)', alpha=0.6)

    # Boundary condition points
    sc1 = plt.scatter(xt_bc[:, 0], xt_bc[:, 1], c=u_bc.ravel(), cmap='jet', marker='x', s=40, label='Boundary condition')
    # Initial condition points
    sc2 = plt.scatter(xt_ic[:, 0], xt_ic[:, 1], c=u_ic.ravel(), cmap='jet', marker='^', s=40, label='Initial condition')
    # Add a single colorbar (optional)
    cbar = plt.colorbar(sc2)
    cbar.set_label("u(x,t)")

    plt.xlabel('x', fontsize=12)
    plt.ylabel('t', fontsize=12)
    plt.title('PDE Domain and PINN Collocation, Boundary, and Initial Condition Points', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_pde_error_domain(residuals, X, T, coll_points_train, coll_points_val):
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-6
    log_residuals = np.log10(residuals + epsilon)

    plt.figure(figsize=(8, 5))

    # Plot the log-scaled residual heatmap
    contour = plt.contourf(X, T, log_residuals, levels=100, cmap='plasma')

    # Add colorbar with custom tick labels
    cbar = plt.colorbar(contour)
    cbar.set_label('Residual')

    # Set ticks (log scale) and format labels as actual residuals
    log_ticks = np.linspace(np.floor(log_residuals.min()), np.floor(log_residuals.max()), 6)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"$10^{{{int(t)}}}$" for t in log_ticks])

    # Overlay collocation points
    plt.scatter(coll_points_train[:, 0], coll_points_train[:, 1], color='black', s=15, edgecolors='white', label='Train Collocation', alpha=0.7)
    plt.scatter(coll_points_val[:, 0], coll_points_val[:, 1], color='orange', edgecolors='white',s=15, label='Val Collocation', alpha=0.7)

    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Log-Scaled PDE Residual with Collocation Points')
    # Dense legend on the right
    plt.legend(
        handlelength=1.5,
        handletextpad=0.5,
        labelspacing=0.3,
        fontsize='medium',
        framealpha=1.0
    )
    plt.tight_layout()
    plt.show()
