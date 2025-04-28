from utils.plotting import PlasmaPlotter

import h5py
import numpy as np

# 1. Create the plotter object
plotter = PlasmaPlotter(save_path="./figs")

# 2. Load your data
frame_idx = 1        # Frame to plot
nx, ny = 128, 89       # Number of points in x and y (from paper description)
points_per_frame = nx * ny

start_idx = frame_idx * points_per_frame
end_idx = (frame_idx + 1) * points_per_frame

with h5py.File("data/plasma_data.h5", "r") as f:
    x_x = f['x_x'][start_idx:end_idx]
    x_y = f['x_y'][start_idx:end_idx]
    y_den = f['y_den'][start_idx:end_idx]
    y_Te = f['y_Te'][start_idx:end_idx]

# 3. Call the plotting function
plotter.plot_plasma_state(
    X0=x_x,
    X1=x_y,
    y_den=y_den,
    y_Te=y_Te,
    N_time=0,        # 0 because we already sliced one frame
    len_skip=1,      # 1 since no skipping in sliced data
    len_2d=points_per_frame
)

# 4. Plot 1d - electric potential 
plotter.plot_1d_potential(
    X0= x_x, X1=x_y, y_plot: List[np.ndarray],
                         len_loop_y=1, inds: np.ndarray)
)