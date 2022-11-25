import h5py
import numpy as np
import matplotlib.pyplot as plt
import include.diagnostics as diag
import matplotlib
plt.rcParams.update({'font.size': 18})
matplotlib.use('TkAgg')

# Load in data:
data_path = 'frames/199f_C-FM=2_third-SQV'   # input('Enter file path of data to view: ')
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

# Calculate spin vectors
fx, fy, fz = diag.calc_spin_vectors(psiP2, psiP1, psi0, psiM1, psiM2)
F = np.sqrt(abs(fx) ** 2 + abs(fy) ** 2 + fz ** 2)

# Calculate spin expectation
spin_expec = tophat * F / n
# spin_expec[n < 1e-6] = 0

# Construct plot at constant y
fig_consty, ax_consty = plt.subplots(1, figsize=(4.2, 3.54))
ax_consty.set_xlim(-x.max(), x.max())
ax_consty.set_ylim(-x.max(), x.max())
ax_consty.set_xlabel(r'$x/\ell$')
ax_consty.set_ylabel(r'$z/\ell$')
y_index = Ny // 2
consty_plot = ax_consty.contourf(X[:, y_index, :], Z[:, y_index, :], spin_expec[:, y_index, :], levels=200, cmap='jet')
constz_cbar = plt.colorbar(consty_plot, ax=ax_consty, pad=0.01)
constz_cbar.set_ticks([0, 1, 2])
constz_cbar.set_ticklabels(['0', '1', '2'])
plt.savefig('../../../plots/spin-2/paper/C-FM=2_third-SQV_spinMag_consty.png', bbox_inches='tight', dpi=200)
plt.show()
