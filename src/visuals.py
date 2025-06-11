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
        sc1 = ax1.scatter(points[:, 0:1], points[:, 1:], c=stored_predictions[str(epoch)], cmap='plasma')
        ax1.set_title('Prediction')
        fig.colorbar(sc1, ax=ax1)
        # Centre subplot - target
        target_high_res = np.load('./data/burgers_sol.npy', allow_pickle=True)[-1]
        sc2 = ax2.scatter(points[:, 0:1], points[:, 1:], c=target_high_res, cmap='plasma')
        ax2.set_title('Target')
        fig.colorbar(sc2, ax=ax2)
        # Right subplot - error
        sc3 = ax3.scatter(points[:, 0:1], points[:, 1:], c=(target_high_res - stored_predictions[str(epoch)]),
                          vmax=0.01, vmin=-0.01, cmap='seismic', alpha=0.75)
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
