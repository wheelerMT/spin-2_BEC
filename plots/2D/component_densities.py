import h5py
import numpy as np
import matplotlib.pyplot as plt
import include.diagnostics as diag
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('TkAgg')


# Load in data:
data_path = input('Enter file path of data to view: ')
data = h5py.File('../../data/2D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Wavefunction
psiP2 = data['wavefunction/psiP2'][:, :, :]
psiP1 = data['wavefunction/psiP1'][:, :, :]
psi0 = data['wavefunction/psi0'][:, :, :]
psiM1 = data['wavefunction/psiM1'][:, :, :]
psiM2 = data['wavefunction/psiM2'][:, :, :]
Wfn = [psiP2, psiP1, psi0, psiM1, psiM2]

total_dens = diag.calc_density(Wfn)
dens_max = np.max(total_dens)

# Grid data:
x, y = data['grid/x'], data['grid/y']
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x[:], y[:], indexing='ij')
Nx, Ny = len(x), len(y)

# Generate figure:
fig = plt.figure(figsize=(15, 4.8))
grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1, 5),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )

for axis in grid:
    axis.set_aspect('equal')

# Axis titles:
axis_titles = [r'$|\psi_2|^2$', r'$|\psi_1|^2$', r'$|\psi_0|^2$', r'$|\psi_{-1}|^2$', r'$|\psi_{-2}|^2$']

# Plot density of last frame of data
for i, axis in enumerate(grid):
    if i == 0:
        axis.set_ylabel(r'$y/\ell$')
    axis.set_xlabel(r'$x/\ell$')

    axis.set_title(axis_titles[i])

    # Plot slice through y-axis:
    plot = axis.contourf(X, Y, abs(Wfn[i][:, :, -1]) ** 2, np.linspace(0, dens_max, 100), cmap='jet')

    if i == len(grid) - 1:
        # Colorbar
        axis.cax.colorbar(plot)
        axis.cax.toggle_label(True)

plt.tight_layout()
plt.show()
