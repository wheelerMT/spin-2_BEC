import h5py
import numpy as np
import matplotlib.pyplot as plt
import include.diagnostics as diag
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('TkAgg')


# Load in data:
data_path = 'frames/25f_C-FM=2_SQV-SQV'  # input('Enter file path of data to view: ')
data = h5py.File('../../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Frame of data to work with
frame = -1

# Wavefunction
psiP2 = data['wavefunction/psiP2'][:, :, :, frame]
psiP1 = data['wavefunction/psiP1'][:, :, :, frame]
psi0 = data['wavefunction/psi0'][:, :, :, frame]
psiM1 = data['wavefunction/psiM1'][:, :, :, frame]
psiM2 = data['wavefunction/psiM2'][:, :, :, frame]

# psiP2 = data['initial_state/psiP2'][:, :, :]
# psiP1 = data['initial_state/psiP1'][:, :, :]
# psi0 = data['initial_state/psi0'][:, :, :]
# psiM1 = data['initial_state/psiM1'][:, :, :]
# psiM2 = data['initial_state/psiM2'][:, :, :]

Wfn = [psiP2, psiP1, psi0, psiM1, psiM2]

total_dens = diag.calc_density(Wfn)
dens_max = np.max(total_dens)

# Grid data:
x, y, z = data['grid/x'], data['grid/y'], data['grid/y']
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = np.meshgrid(x[:], y[:], z[:], indexing='ij')
Nx, Ny, Nz = len(x), len(y), len(z)

# Generate figure:
fig = plt.figure(figsize=(10, 6.4))
grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2, 5),
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
    # Plot slice through z-axis:
    if i <= 4:
        if i == 0:
            axis.set_ylabel(r'$y/\ell$')
        axis.set_xlabel(r'$x/\ell$')
        axis.set_title(axis_titles[i])

        plot = axis.contourf(X[:, :, Nz // 2 + 10], Y[:, :, Nz // 2 + 10], abs(Wfn[i][:, :, Nz // 2 + 10]) ** 2,
                             np.linspace(0, dens_max, 100), cmap='jet')

    if i > 4:
        if i == 5:
            axis.set_ylabel(r'$y/\ell$')
        axis.set_xlabel(r'$x/\ell$')

        plot = axis.contourf(X[:, :, Nz // 2 - 10], Y[:, :, Nz // 2 - 10], abs(Wfn[i - 5][:, :, Nz // 2 - 10]) ** 2,
                             np.linspace(0, dens_max, 100), cmap='jet')

        if i == len(grid) - 1:
            # Colorbar
            axis.cax.colorbar(plot, format='%.0e')
            axis.cax.toggle_label(True)

plt.tight_layout()
plt.savefig('../../../plots/spin-2/talk/C-FM=2_dens.png', bbox_inches="tight")
plt.show()