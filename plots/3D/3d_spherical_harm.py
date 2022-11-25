import h5py
import numpy as np
import matplotlib.pyplot as plt
import include.diagnostics as diag
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib

matplotlib.use('TkAgg')

# Load in data:
data_path = 'C-FM=2_third-SQV'  # input('Enter file path of data to view: ')
data = h5py.File('../../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Frame of data to work with
frame = -1

# Wavefunction
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

# Spherical harmonic grid points:
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 50)
theta, phi = np.meshgrid(theta, phi, indexing='ij')

# Total density
n = diag.calc_density(Wfn)

# Calculate normalised wavefunction
Zeta = diag.normalise_wfn(Wfn)

# Calculate spin-singlet trio
a30 = diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

# Calculate spin vectors
fp, fz = diag.calc_spin_vectors(psiP2, psiP1, psi0, psiM1, psiM2)
F = np.sqrt(fz ** 2 + abs(fp) ** 2)

# Calculate spin expectation
spin_mag = F / n
spin_mag[n < 1e-6] = 0

Xplot, Yplot = np.meshgrid(x, y)
spin_mag = spin_mag[:, Ny // 2, :]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
surf = ax.plot_surface(Xplot, Yplot, spin_mag, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Spherical Harmonics
Y_2p2 = 0.25 * np.sqrt(15 / 2 * np.pi) * np.exp(-2j * phi) * np.sin(theta) ** 2
Y_2p1 = 0.5 * np.sqrt(15 / 2 * np.pi) * np.exp(-1j * phi) * np.sin(theta) * np.cos(theta)
Y_2p0 = 0.25 * np.sqrt(5 / np.pi) * (3 * np.cos(theta) ** 2 - 1)
Y_2m1 = -0.5 * np.sqrt(15 / 2 * np.pi) * np.exp(1j * phi) * np.sin(theta) * np.cos(theta)
Y_2m2 = 0.25 * np.sqrt(15 / 2 * np.pi) * np.exp(2j * phi) * np.sin(theta) ** 2

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for i in range(Nx // 2 - 10, Nx // 2 + 10, 8):
    for j in range(Ny // 2 - 10, Ny // 2 + 10, 8):
        for k in range(Nz // 2 - 10, Nz // 2 + 10, 8):
            zsph = Zeta[0][i, j, k] * Y_2p2 + Zeta[1][i, j, k] * Y_2p1 + Zeta[2][i, j, k] * Y_2p0 + \
                   Zeta[3][i, j, k] * Y_2m1 + Zeta[4][i, j, k] * Y_2m2
            zsph *= 0.5
            xx = 0.5 * abs(zsph) ** 2 * np.sin(theta) * np.cos(phi) + X[i, j, k]
            yy = 0.5 * abs(zsph) ** 2 * np.sin(theta) * np.sin(phi) + Y[i, j, k]
            zz = 0.5 * abs(zsph) ** 2 * np.cos(theta) + Z[i, j, k]

            color_map = cm.binary
            scalarMap = cm.ScalarMappable(norm=Normalize(vmin=-np.pi, vmax=np.pi), cmap=color_map)

            # outputs an array where each C value is replaced with a corresponding color value
            C_colored = scalarMap.to_rgba(np.angle(zsph))
            # ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=C_colored, antialiased=False)
            ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=C_colored, antialiased=False)

plt.show()
