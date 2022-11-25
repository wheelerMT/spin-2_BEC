import h5py
import numpy as np
import include.diagnostics as diag
import plotly.graph_objects as go

# Load in data:
data_path = 'UN-BN_interface_pi2'  # input('Enter file path of data to view: ')
data = h5py.File('../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Frame of data to work with
frame = -1

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

# Spherical harmonic grid points:
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 50)
theta, phi = np.meshgrid(theta, phi, indexing='ij')

# Calculate normalised wavefunction
Zeta = diag.normalise_wfn(Wfn)

# Spherical Harmonics
Y_2p2 = 0.25 * np.sqrt(15 / 2 * np.pi) * np.exp(-2j * phi) * np.sin(theta) ** 2
Y_2p1 = 0.5 * np.sqrt(15 / 2 * np.pi) * np.exp(-1j * phi) * np.sin(theta) * np.cos(theta)
Y_2p0 = 0.25 * np.sqrt(5 / np.pi) * (3 * np.cos(theta) ** 2 - 1)
Y_2m1 = -0.5 * np.sqrt(15 / 2 * np.pi) * np.exp(1j * phi) * np.sin(theta) * np.cos(theta)
Y_2m2 = 0.25 * np.sqrt(15 / 2 * np.pi) * np.exp(2j * phi) * np.sin(theta) ** 2

# * Plot spherical harmonics on a line of wfn where a_30 varies
# Indices used for slicing:
y_index = Ny // 2

# Indices for ranges
x_range_low = Nx // 2 - 6
x_range_high = Nx // 2 + 6
z_range_low = Nz // 2 - 12
z_range_high = Nz // 2 + 12

fig = go.Figure()
fig.update_layout(scene=dict(yaxis=dict(nticks=4, range=[y[y_index] - 1, y[y_index] + 1], showbackground=False),
                  xaxis=dict(showbackground=False), zaxis=dict(showbackground=False)))

for i in range(x_range_low, x_range_high + 1, 2):
    for j in range(z_range_low, z_range_high + 1, 1):
        zsph = Zeta[0][i, y_index, j] * Y_2p2 + Zeta[1][i, y_index, j] * Y_2p1 \
               + Zeta[2][i, y_index, j] * Y_2p0 + Zeta[3][i, y_index, j] * Y_2m1 + Zeta[4][i, y_index, j] * Y_2m2
        scale = 0.4 ** 2
        zsph *= scale
        xx = abs(zsph) ** 2 * np.sin(theta) * np.cos(phi) + X[i, y_index, j]
        yy = abs(zsph) ** 2 * np.sin(theta) * np.sin(phi) + Y[i, y_index, j]
        zz = abs(zsph) ** 2 * np.cos(theta) + Z[i, y_index, j]

        if i == x_range_high & j == z_range_high:
            fig.add_trace(go.Surface(x=xx, y=yy, z=zz, surfacecolor=np.angle(zsph), colorscale="Jet",
                                     colorbar=dict(x=0.75, nticks=10, tickmode="array", tickvals=[-np.pi, 0, np.pi],
                                     tickfont=dict(size=28))))
        else:
            fig.add_trace(go.Surface(x=xx, y=yy, z=zz, surfacecolor=np.angle(zsph), colorscale="Jet", showscale=False))

# camera = dict(
#     eye=dict(x=0., y=-0.8, z=1)
# )
# fig.update_layout(scene_camera=camera)
# fig.write_image("../../data/plots/presentation/{}_cyclicSide.pdf".format(data_path))
fig.show()
