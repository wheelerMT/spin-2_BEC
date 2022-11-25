import h5py
import numpy as np
# import mayavi.mlab as ml
import matplotlib.pyplot as plt
import include.diagnostics as diag


# @ml.animate(delay=10)
# def anim():
#     for j in range(num_of_frames):
#         mlab_plot.mlab_source.set(scalars=abs(a20_frames[j][Nx//2:, :, :]) ** 2)
#         ml.savefig("data/plots/anim%03d.png" % j)
#         yield


Nx, Ny, Nz = 64, 64, 64  # Number of grid points
Mx, My, Mz = Nx // 2, Ny // 2, Nz // 2
dx, dy, dz = 0.5, 0.5, 0.5  # Grid spacing
dkx, dky, dkz = np.pi / (Mx * dx), np.pi / (My * dy), np.pi / (Mz * dz)  # K-space spacing
len_x, len_y, len_z = Nx * dx, Ny * dy, Nz * dz  # Box length

# Generate 1-D spatial grids:
x = np.arange(-Mx, Mx) * dx
y = np.arange(-My, My) * dy
z = np.arange(-Mz, Mz) * dz
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Load in data:
data = h5py.File('data/C-FM_interface.hdf5', 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

psiP2 = data['wavefunction/psiP2'][:, :, :, -1]
psiP1 = data['wavefunction/psiP1'][:, :, :, -1]
psi0 = data['wavefunction/psi0'][:, :, :, -1]
psiM1 = data['wavefunction/psiM1'][:, :, :, -1]
psiM2 = data['wavefunction/psiM2'][:, :, :, -1]
Wfn = [psiP2, psiP1, psi0, psiM1, psiM2]

# # Plot component densities:
# fig, ax = plt.subplots(5, figsize=(6.4, 15))
# for i, wfn in enumerate(Wfn):
#     ax[i].pcolormesh(abs(wfn[:, :, Nx // 2]) ** 2, cmap='jet', vmin=0)
# plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Initial frame diagnostics
# ---------------------------------------------------------------------------------------------------------------------
# Calculate density
n = diag.calc_density(Wfn)

# Calculate normalised wavefunction
Zeta = diag.normalise_wfn(Wfn)

# Calculate spin vectors
fp, fz = diag.calc_spin_vectors(psiP2, psiP1, psi0, psiM1, psiM2)
F = np.sqrt(fz ** 2 + abs(fp) ** 2)

# Calculate spin expectation
spin_expec = F / n
spin_expec[n < 1e-6] = 0

# Calculate spin-singlet terms
a20 = diag.calc_spin_singlet_duo(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

a30 = diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

plt.contourf(X[:, 32, :], Z[:, 32, :], abs(a30[:, 32, :]) ** 2, levels=100)
plt.colorbar()
plt.show()
exit()
r"""
fig = ml.figure(size=(1920, 1080))
mlab_plot = ml.contour3d(abs(a20[Nx//2:, :, :]) ** 2, transparent=False, vmin=0, contours=100)
ml.colorbar(orientation='vertical')
ml.view(azimuth=180, elevation=90, distance=130)  # View along positive x-axis
ml.orientation_axes()



spin_expec_frames = []
a20_frames = []
a30_frames = []

for i in range(num_of_frames):
    # Load in wfn frame by frame
    psiP2 = data['wavefunction/psiP2'][:, :, :, i]
    psiP1 = data['wavefunction/psiP1'][:, :, :, i]
    psi0 = data['wavefunction/psi0'][:, :, :, i]
    psiM1 = data['wavefunction/psiM1'][:, :, :, i]
    psiM2 = data['wavefunction/psiM2'][:, :, :, i]
    Wfn = [psiP2, psiP1, psi0, psiM1, psiM2]

    # Calculate spin vectors
    # fp, fz = diag.calc_spin_vectors(psiP2, psiP1, psi0, psiM1, psiM2)
    # F = np.sqrt(fz ** 2 + abs(fp) ** 2)
    # dens = diag.calc_density([psiP2, psiP1, psi0, psiM1, psiM2])
    #
    # spin_expec = F / n
    # spin_expec[n < 1e-6] = 0
    # spin_expec_frames.append(spin_expec)
    Zeta = diag.normalise_wfn(Wfn)
    a20_frames.append(diag.calc_spin_singlet_duo(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4]))
    # print(np.max(abs(a30_frames[i])) ** 2)

for j in range(num_of_frames):
    mlab_plot.mlab_source.set(scalars=abs(a20_frames[j][Nx // 2:, :, :]) ** 2)
    ml.savefig("data/plots/images/anim%03d.png" % j)

# anim()
# ml.show()


# Plot 2D contours of diagnostics:
fig, ax = plt.subplots(3, figsize=(4.8, 15))
A20_plot = ax[0].contourf(abs(a20[:, :, Nx // 2]) ** 2, levels=100)
plt.colorbar(A20_plot, ax=ax[0])
ax[0].set_title(r'$|A_{20}|^2$')

A30_plot = ax[1].contourf(abs(a30[:, :, Nx // 2]) ** 2, levels=100)
plt.colorbar(A30_plot, ax=ax[1])
ax[1].set_title(r'$|A_{30}|^2$')

F_plot = ax[2].contourf(abs(F[:, :, Nx // 2]), cmap='PuRd', levels=100)
plt.colorbar(F_plot, ax=ax[2])
ax[2].set_title(r'$|\vec{F}|$')

plt.show()
exit()

mlab_plot = ml.contour3d(abs(a20[:, :, :]), transparent=True, vmin=0, vmax=2, contours=100)
ml.colorbar()
ml.show()
"""
