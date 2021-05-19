import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load in data:
data_path = input('Enter file path of data to view: ')
data = h5py.File('../../data/2D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Wavefunction
psiP2 = data['wavefunction/psiP2']
psiP1 = data['wavefunction/psiP1']
psi0 = data['wavefunction/psi0']
psiM1 = data['wavefunction/psiM1']
psiM2 = data['wavefunction/psiM2']
Wfn = [psiP2, psiP1, psi0, psiM1, psiM2]

# Grid data:
x, y = data['grid/x'], data['grid/y']
dx, dy = x[1] - x[0], y[1] - y[0]
X, Y = np.meshgrid(x[:], y[:], indexing='ij')
Nx, Ny = len(x), len(y)

# Generate figure:
fig = plt.figure()
ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
axes = [ax1, ax2, ax3, ax4, ax5]
for axis in axes:
    axis.set_aspect('equal')

# Axis titles:
axis_titles = [r'$|\psi_2|^2$', r'$|\psi_1|^2$', r'$|\psi_0|^2$', r'$|\psi_{-1}|^2$', r'$|\psi_{-2}|^2$']

# Plot density of last frame of data
for i, axis in enumerate(axes):
    axis.set_xlabel(r'$x/\ell$')
    axis.set_ylabel(r'$y/\ell$')
    axis.set_title(axis_titles[i])
    axis.contourf(X, Y, abs(Wfn[i][:, :, -1]) ** 2, np.linspace(0, 0.003, 100))

plt.tight_layout()
plt.show()
