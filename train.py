import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from src.loss import mass, k, mu, loss_pinn
from src.solver import Solver

w_0 = np.sqrt(k/mass)  # ω₀ (angular frequency)
delta = mu/(2*mass)  # δ (delta), the damping factor
w = np.sqrt(w_0**2 - delta**2)  # ω damped angular frequency


def generate_solution_osc(x):
    y = np.exp(-mu/(2*mass)*x) * (np.cos(w*x) + mu/(2*mass*w)*np.sin(w*x))
    return y


def inference_model_and_performance(model, time, analytical_solution):
    # Generate high resolution learned solutions
    test_t_tensor = torch.tensor(time, dtype=torch.float32).reshape(-1, 1)
    # set model to evaluate
    model.eval()
    predicted_solution = model(test_t_tensor).ravel().detach().numpy()
    # compute MAE error
    error = np.abs(analytical_solution - predicted_solution)
    mean_error = np.mean(error)
    return predicted_solution, mean_error


if __name__ == '__main__':

    batch_size = 128

    # Simulate real world scenario by random sampling data
    range_time_ode = 1
    num_sampled_t = 25
    #sampled_t = (np.random.rand(num_sampled_t) * range_time_ode / 2).reshape(-1, 1)
    collocation_points = np.linspace(0, range_time_ode, num_sampled_t)
    sampled_u = generate_solution_osc(collocation_points)

    # Split the data train, val (80-20%)
    train_t, valid_t, train_u, valid_u = train_test_split(collocation_points, sampled_u, test_size=0.2, random_state=52)

    # Convert data to PyTorch tensors
    train_t = torch.tensor(train_t, dtype=torch.float32)
    train_u = torch.tensor(train_u, dtype=torch.float32)
    valid_t = torch.tensor(valid_t, dtype=torch.float32)
    valid_u = torch.tensor(valid_u, dtype=torch.float32)
    # Create TensorDatasets
    train_dataset = TensorDataset(train_t, train_u)
    valid_dataset = TensorDataset(valid_t, valid_u)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    neural_network = torch.nn.Sequential(torch.nn.Linear(1, 32),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(32, 32),
                                        torch.nn.Tanh(),
                                        torch.nn.Linear(32, 1))

    # Hyper-parameters
    epochs = 15000
    lr = 1e-3
    optm = torch.optim.Adam
    # Regularization techniques
    early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min')
    # Model setup
    model = Solver(neural_network, criterion=loss_pinn, lr=lr, optimizer=optm)
    # Trainer setup
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         max_epochs=epochs,
                         callbacks=[early_stopping], check_val_every_n_epoch=100)
    # Training the model
    trainer.fit(model, train_loader, valid_loader)
    # Save the model
    torch.save(model.state_dict(), './model_pinn.pth')

    # Do inference of trained model
    # Generate analytical solution
    analytical_time = np.linspace(0, range_time_ode, 500)  # Creating a tensor of 500 points between 0 and 1
    analytical_solution = generate_solution_osc(
        analytical_time)  # Calculating the corresponding y-values using the oscillator function

    predicted_solution, error = inference_model_and_performance(model, analytical_time, analytical_solution,)

    # Plot the exact solution and training points
    plt.figure(figsize=(10, 6))
    plt.scatter(0, 1, color='green', label='Initial Condition')
    plt.scatter(train_t, len(train_t) * [-0.6], color='green', label='CP train', s=20)
    plt.scatter(valid_t, len(valid_t) * [-0.6], color='red', label='CP valid', s=20)

    plt.plot(analytical_time, analytical_solution, label='Analylical Solution', color='blue', alpha=0.5)
    plt.plot(analytical_time, predicted_solution, color='black', label='Learned Model', alpha=0.75)

    plt.fill_between(analytical_time, analytical_solution, predicted_solution, color='red', alpha=0.1,
                     label=f'Error mae: {round(error, 4)}')

    plt.title('Analytical Solution vs Learned Model PINN')
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.legend()
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()