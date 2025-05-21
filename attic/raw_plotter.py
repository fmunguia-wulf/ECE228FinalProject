from utils.plotting import PlasmaPlotter
from utils.constants import *
import h5py
import numpy as np
import matplotlib.pyplot as plt

SCATTER_POINT_SIZE = 10

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
