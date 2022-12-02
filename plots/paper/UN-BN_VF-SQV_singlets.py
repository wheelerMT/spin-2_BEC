import h5py
import numpy as np
import matplotlib.pyplot as plt
import include.diagnostics as diag
import matplotlib
plt.rcParams.update({'font.size': 18})
matplotlib.use('TkAgg')

# Load in data:
data_path = 'frames/imag_time/49f_UN-BN_VF-SQV'   # input('Enter file path of data to view: ')
data = h5py.File('../../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

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
x, y, z = data['grid/x'][...], data['grid/y'][...], data['grid/y'][...]
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = np.meshgrid(x[:], y[:], z[:], indexing='ij')
Nx, Ny, Nz = len(x), len(y), len(z)

# Tophat for smoothing plots
sigma = 0.5
c0 = 1.32e4
c2 = 146
g = c0 + 4 * c2
Rtf = (15 * g / (4 * np.pi)) ** 0.2
tophat = 0.5 * (1 - np.tanh(sigma * (X ** 2 + Y ** 2 + Z ** 2 - Rtf ** 2)))

# Total density
n = diag.calc_density(Wfn)

# Calculate normalised wavefunction
Zeta = diag.normalise_wfn(Wfn)

# Calculate spin expectation
a20 = tophat * abs(diag.calc_spin_singlet_duo(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])) ** 2
a30 = tophat * abs(diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])) ** 2

# Plot
fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey='row')
for axis in ax:
    axis.set_xlim(-x.max() + 2, x.max() - 2)
    axis.set_ylim(-y.max() + 2, y.max() - 2)
    axis.set_xlabel(r'$y/\ell$')
ax[0].set_ylabel(r'$z/\ell$')
x_index = Ny // 2 + 9

consty_plot = ax[0].contourf(Y[x_index, :, :], Z[x_index, :, :], a20[x_index, :, :], levels=200, cmap='jet')
ax[0].plot([-7, 7], [0, 0], 'w--', linewidth=3)
constz_cbar = plt.colorbar(consty_plot, ax=ax[0], pad=0.01)
constz_cbar.set_ticks([0, 0.2])
constz_cbar.set_ticklabels(['0', '0.2'])

plot = ax[1].contourf(Y[x_index, :, :], Z[x_index, :, :], a30[x_index, :, :], levels=200, cmap='jet')
ax[1].plot([-7, 7], [0, 0], 'w--', linewidth=3)
cbar = plt.colorbar(plot, ax=ax[1], pad=0.01)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['0', '1', '2'])

plt.tight_layout()
plt.savefig('../../../plots/spin-2/paper/UN-BN_VF-SQV_spinSinglets.png', bbox_inches='tight', dpi=200)
plt.show()
