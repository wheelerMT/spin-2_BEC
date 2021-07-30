import h5py
import numpy as np
import include.diagnostics as diag
import plotly.graph_objects as go

# Load in data:
data_path = input('Enter file path of data to view: ')
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

# a30
a30 = diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

# Spherical Harmonics
Y_2p2 = 0.25 * np.sqrt(15 / 2 * np.pi) * np.exp(-2j * phi) * np.sin(theta) ** 2
Y_2p1 = 0.5 * np.sqrt(15 / 2 * np.pi) * np.exp(-1j * phi) * np.sin(theta) * np.cos(theta)
Y_2p0 = 0.25 * np.sqrt(5 / np.pi) * (3 * np.cos(theta) ** 2 - 1)
Y_2m1 = -0.5 * np.sqrt(15 / 2 * np.pi) * np.exp(1j * phi) * np.sin(theta) * np.cos(theta)
Y_2m2 = 0.25 * np.sqrt(15 / 2 * np.pi) * np.exp(2j * phi) * np.sin(theta) ** 2

# Indices used for slicing:
z_index = Nz // 2 + 10

fig = go.Figure()
fig.update_layout(scene=dict(zaxis=dict(nticks=4, range=[z[z_index] - 1, z[z_index] + 1])))

for i in range(Nx // 2 - 5, Nx // 2 + 6, 1):
    for j in range(Ny // 2 - 5, Ny // 2 + 6, 2):
        zsph = Zeta[0][i, j, z_index] * Y_2p2 + Zeta[1][i, j, z_index] * Y_2p1 + Zeta[2][
            i, j, z_index] * Y_2p0 + \
               Zeta[3][i, j, z_index] * Y_2m1 + Zeta[4][i, j, z_index] * Y_2m2
        zsph *= 0.5
        xx = 0.5 * abs(zsph) ** 2 * np.sin(theta) * np.cos(phi) + X[i, j, z_index]
        yy = 0.5 * abs(zsph) ** 2 * np.sin(theta) * np.sin(phi) + Y[i, j, z_index]
        zz = 0.5 * abs(zsph) ** 2 * np.cos(theta) + Z[i, j, z_index]

        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, surfacecolor=np.angle(zsph)))

fig.show()
