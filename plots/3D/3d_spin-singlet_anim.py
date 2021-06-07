import h5py
import numpy as np
import matplotlib.pyplot as plt
import include.diagnostics as diag
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.animation as animation
import matplotlib

matplotlib.use('TkAgg')

# Load in data:
data_path = input('Enter file path of data to view: ')
data = h5py.File('../../data/3D/{}.hdf5'.format(data_path), 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]
print("Working with {} frames of data".format(num_of_frames))

# Wavefunction
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

# Total density
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

# Generate figure:
fig = plt.figure(figsize=(12, 4.8))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.4, share_all=True, cbar_location="right", cbar_mode="each",
                 cbar_size="7%", cbar_pad=0.05)

for axis in grid:
    axis.set_aspect('equal')

# Plot titles:
grid[0].set_title(r'$|<\vec{F}>|$')
grid[1].set_title(r'$|A_{20}|^2$')
grid[2].set_title(r'$|A_{30}|^2$')

# Plot axis labels:
grid[0].set_ylabel(r'$z/\ell$')
for axis in grid:
    axis.set_xlabel(r'$x/\ell$')

# Plot diagnostics
spin_plot = grid[0].contourf(X[:, Nx // 2, :], Z[:, Nx // 2, :], abs(spin_expec[:, Nx // 2, :, 0]),
                             np.linspace(0, 2, 100), cmap='jet')
grid[0].plot([-6, 6], [0, 0], 'w--')
a20_plot = grid[1].contourf(X[:, Nx // 2, :], Z[:, Nx // 2, :], abs(a20[:, Nx // 2, :, 0]) ** 2,
                            np.linspace(0, 1 / 5, 100), cmap='jet')
grid[1].plot([-6, 6], [0, 0], 'w--')
a30_plot = grid[2].contourf(X[:, Nx // 2, :], Z[:, Nx // 2, :], abs(a30[:, Nx // 2, :, 0]) ** 2, np.linspace(0, 2, 100),
                            cmap='jet')
grid[2].plot([-6, 6], [0, 0], 'w--')

# Set colorbars
grid[0].cax.colorbar(spin_plot, ticks=[0, 1, 2])
grid[0].cax.toggle_label(True)
grid[1].cax.colorbar(a20_plot, ticks=[0, 1 / 5])
grid[1].cax.toggle_label(True)
grid[2].cax.colorbar(a30_plot, ticks=[0, 1, 2])
grid[2].cax.toggle_label(True)


# Animation function
def animate(i):
    for contour in grid:
        for c in contour.collections:
            c.remove()

    grid[0].contourf(X[:, Nx // 2, :], Z[:, Nx // 2, :], abs(spin_expec[:, Nx // 2, :, i]), np.linspace(0, 2, 100),
                     cmap='jet')
    grid[0].plot([-6, 6], [0, 0], 'w--')
    grid[1].contourf(X[:, Nx // 2, :], Z[:, Nx // 2, :], abs(a20[:, Nx // 2, :, i]) ** 2,
                     np.linspace(0, 1 / 5, 100), cmap='jet')
    grid[1].plot([-6, 6], [0, 0], 'w--')
    grid[2].contourf(X[:, Nx // 2, :], Z[:, Nx // 2, :], abs(a30[:, Nx // 2, :, i]) ** 2,
                     np.linspace(0, 2, 100), cmap='jet')
    grid[2].plot([-6, 6], [0, 0], 'w--')

    cont_dens = [grid[0], grid[1], grid[2]]
    print('On density iteration %i' % (i + 1))
    plt.suptitle(r'$\tau$ = %2f' % (1 / 2 * i))
    return cont_dens


# Calls the animation function and saves the result
anim = animation.FuncAnimation(fig, animate, frames=num_of_frames, repeat=False)
anim.save('../../data/plots/{}.mp4'.format(data_path), dpi=200, writer=animation.FFMpegWriter(fps=2))
print('Density video saved successfully.')
