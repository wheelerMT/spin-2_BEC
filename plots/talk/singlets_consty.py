import h5py
import numpy as np
import matplotlib.pyplot as plt
import include.diagnostics as diag
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib

matplotlib.use('TkAgg')

# Load in data:
data_path = 'C-FM=2_SQV-SQV'   # input('Enter file path of data to view: ')
data = h5py.File('../../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Frame of data to work with
frame = 0

# Wavefunction
# psiP2 = data['wavefunction/psiP2'][:, :, :, frame]
# psiP1 = data['wavefunction/psiP1'][:, :, :, frame]
# psi0 = data['wavefunction/psi0'][:, :, :, frame]
# psiM1 = data['wavefunction/psiM1'][:, :, :, frame]
# psiM2 = data['wavefunction/psiM2'][:, :, :, frame]

psiP2 = data['initial_state/psiP2'][:, :, :]
psiP1 = data['initial_state/psiP1'][:, :, :]
psi0 = data['initial_state/psi0'][:, :, :]
psiM1 = data['initial_state/psiM1'][:, :, :]
psiM2 = data['initial_state/psiM2'][:, :, :]

Wfn = [psiP2, psiP1, psi0, psiM1, psiM2]

# Grid data:
x, y, z = data['grid/x'], data['grid/y'], data['grid/y']
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = np.meshgrid(x[:], y[:], z[:], indexing='ij')
Nx, Ny, Nz = len(x), len(y), len(z)

# Total density
n = diag.calc_density(Wfn)

# Calculate normalised wavefunction
Zeta = diag.normalise_wfn(Wfn)

# Calculate spin vectors
fx, fy, fz = diag.calc_spin_vectors(psiP2, psiP1, psi0, psiM1, psiM2)
F = np.sqrt(abs(fx) ** 2 + abs(fy) ** 2 + fz ** 2)

# Calculate spin expectation
spin_expec = F / n
spin_expec[n < 1e-6] = 0

# Calculate spin-singlet terms
a20 = diag.calc_spin_singlet_duo(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

a30 = diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

# Generate figure:
fig = plt.figure(figsize=(4.5, 2.5))
grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                 nrows_ncols=(1, 2),
                 axes_pad=0.4,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="each",
                 cbar_size="7%",
                 cbar_pad=0.05,
                 )

for axis in grid:
    axis.set_aspect('equal')

grid[0].set_xlim(-10, 10)
grid[0].set_ylim(-10, 10)

# y and z indices for graphs:
y_ind = Ny // 2
z_upper_ind = Nz // 2 + 10
z_lower_ind = Nz // 2 - 10

# Plot titles:
grid[0].set_title(r'$|<\vec{F}>|$')
# grid[0].set_title(r'$|A_{20}|^2$')
grid[1].set_title(r'$|A_{30}|^2$')

# Plot axis labels:
grid[0].set_ylabel(r'$z/\ell$')
for axis in grid:
    axis.set_xlabel(r'$x/\ell$')

# ------------------------------------------------------------------
# Plot diagnostics
# ------------------------------------------------------------------
# First row:
one_spin_plot = grid[0].contourf(Y[y_ind, :, :], Z[y_ind, :, :], abs(spin_expec[:, y_ind, :]),
                                 np.linspace(0, 2.01, 100), cmap='jet')
# one_a20_plot = grid[0].contourf(X[:, y_ind, :], Z[:, y_ind, :], abs(a20[:, y_ind, :]) ** 2,
#                                 np.linspace(0, 0.201, 100), cmap='jet')
one_a30_plot = grid[1].contourf(X[:, y_ind, :], Z[:, y_ind, :], abs(a30[:, y_ind, :]) ** 2,
                                np.linspace(0, 2.01, 100), cmap='jet')

# Set colorbars
grid[0].cax.colorbar(one_spin_plot, ticks=[0, 1, 2])
grid[0].cax.toggle_label(True)
# grid[0].cax.colorbar(one_a20_plot, ticks=[0, 1 / 5])
# grid[0].cax.toggle_label(True)
grid[1].cax.colorbar(one_a30_plot, ticks=[0, 2])
grid[1].cax.toggle_label(True)

plt.tight_layout()
plt.savefig('../../../plots/spin-2/write-up/C-FM=2_SQV-SQV_initial.png', dpi=200, bbox_inches="tight")
plt.show()
