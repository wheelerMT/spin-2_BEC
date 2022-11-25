import h5py
import numpy as np
import include.diagnostics as diag
from scipy.special import sph_harm
from mayavi import mlab

# Load in data file:
data_path = 'frames/303f_C-FM=2_SQV-SQV'  # input('Enter file path of data to view: ')
data = h5py.File('../../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Load in wavefunction data
initial_state = False
if initial_state:
    psiP2 = data['initial_state/psiP2'][:, :, :]
    psiP1 = data['initial_state/psiP1'][:, :, :]
    psi0 = data['initial_state/psi0'][:, :, :]
    psiM1 = data['initial_state/psiM1'][:, :, :]
    psiM2 = data['initial_state/psiM2'][:, :, :]
else:
    frame = 1
    psiP2 = data['wavefunction/psiP2'][:, :, :, frame]
    psiP1 = data['wavefunction/psiP1'][:, :, :, frame]
    psi0 = data['wavefunction/psi0'][:, :, :, frame]
    psiM1 = data['wavefunction/psiM1'][:, :, :, frame]
    psiM2 = data['wavefunction/psiM2'][:, :, :, frame]

Wfn = [psiP2, psiP1, psi0, psiM1, psiM2]
Zeta = diag.normalise_wfn(Wfn)

# Grid data:
x, y, z = data['grid/x'], data['grid/y'], data['grid/y']
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = np.meshgrid(x[:], y[:], z[:], indexing='ij')
Nx, Ny, Nz = len(x), len(y), len(z)

# Generate theta and phi grids
theta = np.linspace(0, 2 * np.pi)
phi = np.linspace(0, np.pi)
Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

# Generate spherical harmonics arrays
Y_2p2 = sph_harm(2, 2, Theta, Phi)
Y_2p1 = sph_harm(1, 2, Theta, Phi)
Y_2p0 = sph_harm(0, 2, Theta, Phi)
Y_2m1 = sph_harm(-1, 2, Theta, Phi)
Y_2m2 = sph_harm(-2, 2, Theta, Phi)

# Set up figure
mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1080, 1080))
scale = 0.25

z_index = Nz // 2 - 20
x_lower_index = Nx // 2 - 8
x_upper_index = Nx // 2 + 8
y_lower_index = Ny // 2 - 8
y_upper_index = Ny // 2 + 8
for i in range(x_lower_index, x_upper_index):
    for j in range(y_lower_index, y_upper_index):
        zsph = Zeta[0][i, j, z_index] * Y_2p2 + Zeta[1][i, j, z_index] * Y_2p1 \
               + Zeta[2][i, j, z_index] * Y_2p0 + Zeta[3][i, j, z_index] * Y_2m1 \
               + Zeta[4][i, j, z_index] * Y_2m2

        xx = scale * abs(zsph) ** 2 * np.sin(Phi) * np.cos(Theta) + X[i, j, z_index]
        yy = scale * abs(zsph) ** 2 * np.sin(Phi) * np.sin(Theta) + Y[i, j, z_index]
        zz = scale * abs(zsph) ** 2 * np.cos(Phi) + Z[i, j, z_index]
        mlab.mesh(xx, yy, zz, scalars=np.angle(zsph), colormap='jet')

mlab.axes(extent=[x[x_lower_index], x[x_upper_index], y[y_lower_index], y[y_upper_index], z[z_index], z[z_index]],
          y_axis_visibility=False)
mlab.xlabel(r'$x/\ell$')
mlab.ylabel(r'$y/\ell$')
cbar = mlab.colorbar(orientation='vertical', nb_labels=0, label_fmt='%0.2f')
mlab.show()
