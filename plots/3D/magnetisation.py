import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load in data file:
data_path = 'UN-BN_SV-SV'  # input('Enter file path of data to view: ')
data = h5py.File('../../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Load wavefunction data
psiP2 = data['wavefunction/psiP2'][:, :, :, :]
psiP1 = data['wavefunction/psiP1'][:, :, :, :]
psi0 = data['wavefunction/psi0'][:, :, :, :]
psiM1 = data['wavefunction/psiM1'][:, :, :, :]
psiM2 = data['wavefunction/psiM2'][:, :, :, :]

Wfn = [psiP2, psiP1, psi0, psiM1, psiM2]

# Grid data:
x, y, z = data['grid/x'], data['grid/y'], data['grid/y']
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = np.meshgrid(x[:], y[:], z[:], indexing='ij')
Nx, Ny, Nz = len(x), len(y), len(z)

# Calculate magnetisation
magnetisation = np.sum(2 * (abs(psiP2) ** 2 - abs(psiM2) ** 2) + abs(psiP1) ** 2 - abs(psiM1) ** 2, axis=(0, 1, 2))
# magnetisation /= (Nx * dx * Ny * dy * Nz * dz)
plt.plot(magnetisation)
# plt.ylim(-1, 1)
plt.show()
