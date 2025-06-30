import numpy as np
import matplotlib.pyplot as plt
Nx, Nt = 200, 300
nu = 0.025
# Target resolution
x_target = np.linspace(-1, 1, Nx)
t_target = np.linspace(0, 1, Nt)

# Fine resolution for stability
Nx_fine = 1024
x_fine = np.linspace(-1, 1, Nx_fine)
dx = x_fine[1] - x_fine[0]

dt = 0.00001
Nt_fine = int(1.0 / dt)
save_times = np.linspace(0, Nt_fine-1, Nt, dtype=int)  # time indices to save

# Initialize fine grid
u = -np.sin(np.pi * x_fine)
u[0] = 0
u[-1] = 0

# Output array (Nt, Nx)
u_all = np.zeros((Nt, Nx))

# Save initial state (interpolated)
u_all[0] = np.interp(x_target, x_fine, u)

frame = 1
for n in range(1, Nt_fine):
    un = u.copy()
    u[1:-1] = (un[1:-1]
               - dt * un[1:-1] * (un[2:] - un[:-2]) / (2 * dx)
               + nu * dt * (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2)
    u[0] = 0
    u[-1] = 0

    if n == save_times[frame]:
        u_all[frame] = np.interp(x_target, x_fine, u)
        frame += 1
        if frame == Nt:
            break

# Final check
print("u_all shape:", u_all.shape)  # (100, 256)
np.save('../u_solution.npy', u_all)


# Optional: plot
plt.imshow(u_all, extent=[-1, 1, 0, 1], origin='lower', aspect='auto', cmap='jet')
plt.xlabel('x')
plt.ylabel('t')
plt.title("Subsampled u(x, t) from fine simulation")
plt.colorbar(label='u')
plt.show()