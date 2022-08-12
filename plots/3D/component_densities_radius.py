import h5py
import numpy as np
import include.diagnostics as diag
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

# Load in data:
data_path = 'frames/10f_C-FM=2_third-SQV'   # input('Enter file path of data to view: ')
data = h5py.File('../../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Frame of data to work with
frame = 1

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

z_index = Nz // 2 + 10

# Calculate spherical sum
centerx = Nx // 2
centery = Ny // 2
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)
nc = np.zeros(box_radius,)  # Counts the number of times we sum over a given shell
plus_two_dens = np.zeros(box_radius, )
minus_one_dens = np.zeros(box_radius, )
for i in range(Nx):
    for j in range(Ny):
        r = int(np.ceil(np.sqrt((i - centerx) ** 2 + (j - centery) ** 2)))
        nc[r] += 1

        plus_two_dens[r] += abs(psiP2[i, j, z_index]) ** 2
        minus_one_dens[r] += abs(psiM1[i, j, z_index]) ** 2

plus_two_dens /= nc
minus_one_dens /= nc

r = np.sqrt(x[...] ** 2 + y[...] ** 2)
plt.plot(r[Nx // 2:Nx // 2 + 30], plus_two_dens[:30], 'r', label=r'$|\psi_2|^2$')
plt.plot(r[Nx // 2:Nx // 2 + 30], minus_one_dens[:30], 'b', label=r'$|\psi_{-1}|^2$')
plt.xlabel(r'$r/\ell$')
plt.ylabel('Density')
plt.xlim(0, 6)
plt.legend()
plt.savefig('../../../plots/spin-2/write-up/component_densities_radius.png', bbox_inches='tight')
plt.show()
