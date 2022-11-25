import numpy as np
import h5py
import matplotlib.pyplot as plt
import include.diagnostics as diag
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams.update({'font.size': 18})

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
x, y, z = data['grid/x'], data['grid/y'], data['grid/y']
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = np.meshgrid(x[:], y[:], z[:], indexing='ij')
Nx, Ny, Nz = len(x), len(y), len(z)
r_numerical = np.sqrt(x[...] ** 2 + y[...] ** 2)
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

# Singlet trio
a30 = diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

z_index = Nz // 2 + 2

# Calculate spherical sum
centerx = Nx // 2
centery = Ny // 2
box_radius = int(np.ceil(np.sqrt(Nx ** 2 + Ny ** 2) / 2) + 1)
nc = np.zeros(box_radius,)  # Counts the number of times we sum over a given shell
spin_expec_r = np.zeros(box_radius, )
a30_r = np.zeros(box_radius, )
for i in range(Nx):
    for j in range(Ny):
        r = int(np.ceil(np.sqrt((i - centerx) ** 2 + (j - centery) ** 2)))
        nc[r] += 1

        spin_expec_r[r] += spin_expec[i, j, z_index]
        a30_r[r] += abs(a30[i, j, z_index]) ** 2

spin_expec_r /= nc
a30_r /= nc

# Define radius
N_pts = 10000
r = np.linspace(0, 10, N_pts)

eta = 3 * np.tanh(r / 3) - 1

# Get Thomas-Fermi radius for these parameters
c0 = 10000
c2 = 2000
g = c0 + 4 * c2
Rtf = (15 * g / (4 * np.pi)) ** 0.2
Tf_dens = 15 / (8 * np.pi * Rtf ** 2) * (1 - r ** 2 / Rtf ** 2)
Tf_dens = np.where(Tf_dens < 0, 0, Tf_dens)

psiP2 = 1 / np.sqrt(3) * np.sqrt((1 + eta))
psiP1 = np.zeros(N_pts, )
psi0 = np.zeros(N_pts, )
psiM1 = 1 / np.sqrt(3) * np.sqrt((2 - eta))
psiM2 = np.zeros(N_pts, )

Zeta = diag.normalise_wfn([psiP2, psiP1, psi0, psiM1, psiM2])
a30 = diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])
fx, fy, fz = diag.calc_spin_vectors(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])
spin_expec = np.sqrt(abs(fx) ** 2 + abs(fy) ** 2 + fz ** 2)

fig, ax = plt.subplots(1, )
# ax.set_xlim(0, 6)
# ax.plot(r, 10 * abs(psiP2) ** 2, 'r', label=r'$|\psi_2|^2$')
# ax.plot(r, 10 * abs(psiM1) ** 2, 'b', label=r'$|\psi_{-1}|^2$')
# ax.set_xlabel(r'$r$')
# ax.set_ylabel('Density')
# ax.legend()

ax.set_xlim(0, 6)
ax.plot(r, spin_expec, 'k', label=r'$|\vec{F}(\vec{r})|$')
ax.plot(r_numerical[Nx//2:Nx//2+28], spin_expec_r[:28], 'k--')
# ax.plot(r, abs(a30) ** 2, 'k--', label=r'$|A_{30}(\vec{r})|^2$')
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'Value')
ax.legend()

# plt.savefig('../../../plots/spin-2/paper/spin_singlet_radius_analytical.pdf', bbox_inches='tight')
plt.show()
