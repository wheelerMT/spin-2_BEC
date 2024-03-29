import h5py
import numpy as np
import matplotlib.pyplot as plt
import include.diagnostics as diag
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('TkAgg')

# Load in data:
data_path = 'frames/240f_UN-BN_SQV-SQV'   # input('Enter file path of data to view: ')
data = h5py.File('../../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))
# print(data['saved_times'][...])

# Frame of data to work with
frame = 0

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

# Tophat for smoothing plots
sigma = 0.5
c0 = 1.32e4
c2 = 146
g = c0 + 4 * c2
Rtf = (15 * g / (4 * np.pi)) ** 0.2
tophat = 0.5 * (1 - np.tanh(sigma * (X ** 2 + Y ** 2 + Z ** 2 - Rtf ** 2)))

# Calculate spin expectation
spin_expec = tophat * F / n

# Calculate spin-singlet terms
a20 = tophat * diag.calc_spin_singlet_duo(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

a30 = tophat * diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

# Generate figure:
fig = plt.figure()
grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                 nrows_ncols=(3, 3),
                 axes_pad=(0.4, 0.15),
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="each",
                 cbar_size="7%",
                 cbar_pad=0.05,
                 )

for axis in grid:
    axis.set_aspect('equal')

# y and z indices for graphs:
y_ind = Ny // 2
z_upper_ind = Nz // 2 + 20
z_lower_ind = Nz // 2 - 20

# Plot titles:
grid[0].set_title(r'$|<\vec{F}>|$')
grid[1].set_title(r'$|A_{20}|^2$')
grid[2].set_title(r'$|A_{30}|^2$')
# grid[3].set_title(r'Z = {}'.format(z[z_upper_ind]), x=0.2, fontsize=10)
# grid[6].set_title(r'Z = {}'.format(z[z_lower_ind]), x=0.2, fontsize=10)

# Plot axis labels:
grid[0].set_ylabel(r'$z/\ell$')
grid[3].set_ylabel(r'$y/\ell$')
grid[6].set_ylabel(r'$y/\ell$')
for axis in grid:
    axis.set_xlabel(r'$x/\ell$')

for axis in [grid[0], grid[1], grid[2]]:
    axis.plot([-6, 6], [0, 0], 'w--')  # Plot white dashed line at z=0

# ------------------------------------------------------------------
# Plot diagnostics
# ------------------------------------------------------------------

# First row:
one_spin_plot = grid[0].contourf(X[:, y_ind, :], Z[:, y_ind, :], abs(spin_expec[:, y_ind, :]),
                                 np.linspace(0, 2.01, 100), cmap='jet')
one_a20_plot = grid[1].contourf(X[:, y_ind, :], Z[:, y_ind, :], abs(a20[:, y_ind, :]) ** 2,
                                np.linspace(0, 0.201, 100), cmap='jet')
one_a30_plot = grid[2].contourf(X[:, y_ind, :], Z[:, y_ind, :], abs(a30[:, y_ind, :]) ** 2,
                                np.linspace(0, 2.01, 100), cmap='jet')

# Second row:
two_spin_plot = grid[3].contourf(X[:, :, z_upper_ind], Y[:, :, z_upper_ind], abs(spin_expec[:, :, z_upper_ind]),
                                 np.linspace(0, 2.01, 100), cmap='jet')
two_a20_plot = grid[4].contourf(X[:, :, z_upper_ind], Y[:, :, z_upper_ind], abs(a20[:, :, z_upper_ind]) ** 2,
                                np.linspace(0, 0.201, 100), cmap='jet')
two_a30_plot = grid[5].contourf(X[:, :, z_upper_ind], Y[:, :, z_upper_ind], abs(a30[:, :, z_upper_ind]) ** 2,
                                np.linspace(0, 2.01, 100), cmap='jet')

# Third row:
three_spin_plot = grid[6].contourf(X[:, :, z_lower_ind], Y[:, :, z_lower_ind], abs(spin_expec[:, :, z_lower_ind]),
                                   np.linspace(0, 2.01, 100), cmap='jet')
three_a20_plot = grid[7].contourf(X[:, :, z_lower_ind], Y[:, :, z_lower_ind], abs(a20[:, :, z_lower_ind]) ** 2,
                                  np.linspace(0, 0.201, 100), cmap='jet')
three_a30_plot = grid[8].contourf(X[:, :, z_lower_ind], Y[:, :, z_lower_ind], abs(a30[:, :, z_lower_ind]) ** 2,
                                  np.linspace(0, 2.01, 100), cmap='jet')

# Set colorbars
for i, contour in enumerate([one_spin_plot, two_spin_plot, three_spin_plot]):
    grid[i * 3].cax.colorbar(contour, ticks=[0, 1, 2])
    grid[i * 3].cax.toggle_label(True)
for i, contour in enumerate([one_a20_plot, two_a20_plot, three_a20_plot]):
    grid[(i * 3) + 1].cax.colorbar(contour, ticks=[0, 1 / 5])
    grid[(i * 3) + 1].cax.toggle_label(True)
for i, contour in enumerate([one_a30_plot, two_a30_plot, three_a30_plot]):
    grid[(i * 3) + 2].cax.colorbar(contour, ticks=[0, 2])
    grid[(i * 3) + 2].cax.toggle_label(True)

plt.tight_layout()
# plt.savefig('../../../plots/spin-2/write-up/C-FM=2_coreless_singlets.png', bbox_inches="tight", dpi=200)
plt.show()
