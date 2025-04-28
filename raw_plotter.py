from utils.plotting import PlasmaPlotter
from utils.constants import *
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

SCATTER_POINT_SIZE = 10

def solve_poisson_sparse(n_e, nx, ny, dx, dy=None):
    """
    Solve Poisson's equation ∇²φ = -(e/ε₀)(n_e - n0) using sparse matrix solve.
    Assumes physical units, and PLASMA_DENSITY as background.
    """
    if dy is None:
        dy = dx

    ELECTRON_CHARGE = 1.60217662e-19  # C
    EPSILON_0 = 8.854187817e-12        # F/m

    # Subtract physical background density
    RHS = -(ELECTRON_CHARGE / EPSILON_0) * (n_e - PLASMA_DENSITY)
    RHS = RHS.flatten()

    N = nx * ny

    main_diag = -2*(1/dx**2 + 1/dy**2) * np.ones(N)
    side_diag = 1/dx**2 * np.ones(N-1)
    for i in range(1, nx):
        side_diag[i*ny-1] = 0
    up_down_diag = 1/dy**2 * np.ones(N-ny)

    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    offsets = [0, -1, 1, -ny, ny]

    A = diags(diagonals, offsets, shape=(N, N), format='csr')
    phi = spsolve(A, RHS)
    return phi.reshape((nx, ny))

def plot_predicted_phi(X0, X1, phi_2d, save_path="./figs", colormap="inferno"):
    """
    Plot the predicted electric potential φ(x,y).
    """
    nx, ny = phi_2d.shape
    X = X0.reshape((nx, ny))
    Y = X1.reshape((nx, ny))

    # Convert to cm
    factor_space = 100.0 * MINOR_RADIUS
    X_cm = factor_space * X
    Y_cm = factor_space * Y

    plt.figure(figsize=(8,6))
    p = plt.pcolormesh(X_cm, Y_cm, phi_2d, shading='auto', cmap=colormap)
    plt.colorbar(p, label=r'$\phi$ (V)')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.title(r'Predicted electric potential: $\phi(x,y)$ (V)')
    plt.tight_layout()
    plt.savefig(f"{save_path}/predicted_phi.png")
    plt.savefig(f"{save_path}/predicted_phi.eps")
    plt.show()

# -------------------------------------------------------------------------------
# 1. Create the plotter object
plotter = PlasmaPlotter(save_path="./figs")

# 2. Load your data
frame_idx = 1
nx, ny = 128, 89
points_per_frame = nx * ny

start_idx = frame_idx * points_per_frame
end_idx = (frame_idx + 1) * points_per_frame

with h5py.File("data/plasma_data.h5", "r") as f:
    x_x = f['x_x'][start_idx:end_idx]
    x_y = f['x_y'][start_idx:end_idx]
    y_den = f['y_den'][start_idx:end_idx]
    y_Te = f['y_Te'][start_idx:end_idx]

print("Density min/max before scaling:", y_den.min(), y_den.max(), y_den.mean())

# 3. Rescale density to physical units
y_den = y_den * PLASMA_DENSITY
print("Density min/max after scaling:", y_den.min(), y_den.max(), y_den.mean())

# 4. Plot plasma state
plotter.plot_plasma_state(
    X0=x_x,
    X1=x_y,
    y_den=y_den,
    y_Te=y_Te,
    N_time=0,
    len_skip=1,
    len_2d=points_per_frame
)

# 5. Solve Poisson equation for electric potential
y_den_2d = y_den.reshape((nx, ny))

# Estimate dx, dy from data
dx = np.mean(np.diff(np.sort(np.unique(x_x))))
dy = np.mean(np.diff(np.sort(np.unique(x_y))))

# Solve
phi_2d = solve_poisson_sparse(y_den_2d, nx, ny, dx, dy)

# 6. Plot predicted φ(x,y)
plot_predicted_phi(x_x, x_y, phi_2d)