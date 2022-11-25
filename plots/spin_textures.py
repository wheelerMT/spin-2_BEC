import h5py
import numpy as np
import matplotlib.pyplot as plt
import include.diagnostics as diag
import matplotlib

matplotlib.use('TkAgg')

# Load in data:
data_path = 'C-FM=2_coreless'  # input('Enter file path of data to view: ')
data = h5py.File('../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Frame of data to work with
frame = 10

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
c0 = data['parameters/c0'][...]
c2 = data['parameters/c2'][...]
scale = 0.8
g = c0 + 4 * c2
Rtf = scale * np.ones((Nx, Ny, Nz)) * (15 * g / (4 * np.pi)) ** 0.2

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

# Set up plot
fig, ax = plt.subplots(1, )
ax.set_xlabel('$x/\ell$')
ax.set_ylabel('$y/\ell$')

# Plot spin magnitude
z_slice = Nz // 2 + 10
extent = X.min(), X.max(), Y.min(), Y.max()  # Axis limits
spin_mag_plot = ax.imshow(spin_expec[:, :, z_slice], vmin=0, vmax=2, cmap='PuRd', extent=extent)

# Generate 2D slices
X, Y = X[:, :, z_slice], Y[:, :, z_slice]
fx, fy = fx[:, :, z_slice], fy[:, :, z_slice]
plottable_points = np.where(np.sqrt(X ** 2 + Y ** 2) < Rtf[:, :, z_slice])

ax.quiver(X[plottable_points], Y[plottable_points], fx[plottable_points], fy[plottable_points], scale=1.)
plt.colorbar(spin_mag_plot)
plt.show()
