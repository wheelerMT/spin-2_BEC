import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import include.diagnostics as diag
from matplotlib import animation


def animate(i):
    print(f'On frame number {i}')
    # Calculate new quantities
    wfn = [psiP2[:, :, z_index, i], psiP1[:, :, z_index, i], psi0[:, :, z_index, i], psiM1[:, :, z_index, i],
           psiM2[:, :, z_index, i]]

    # Total density
    n = diag.calc_density(wfn)

    # Calculate normalised wavefunction
    Zeta = diag.normalise_wfn(wfn)

    # Calculate spin vectors
    fx, fy, fz = diag.calc_spin_vectors(wfn[0], wfn[1], wfn[2], wfn[3], wfn[4])
    F = np.sqrt(abs(fx) ** 2 + abs(fy) ** 2 + fz ** 2)

    # Calculate spin expectation
    spin_expec = F / n
    spin_expec[n < 1e-6] = 0

    # Calculate spin-singlet terms
    a20 = diag.calc_spin_singlet_duo(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

    a30 = diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

    spin_plot.set_data(abs(spin_expec[:, :]))
    a20_plot.set_data(abs(a20[:, :]) ** 2)
    a30_plot.set_data(abs(a30[:, :]) ** 2)

    plt.suptitle(r'$\bar{t} = $' + f'{Nframe * dt * i}', y=0.7)


# Load in data
filename = 'UN-BN_SQV-SQV'
data = h5py.File(f'../data/interface/{filename}.hdf5', 'r')
num_of_frames = data['wavefunction/psiP2'].shape[-1]

# Wavefunction
psiP2 = data['wavefunction/psiP2']
psiP1 = data['wavefunction/psiP1']
psi0 = data['wavefunction/psi0']
psiM1 = data['wavefunction/psiM1']
psiM2 = data['wavefunction/psiM2']

# Grid data:
x, y, z = data['grid/x'], data['grid/y'], data['grid/y']
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = np.meshgrid(x[:], y[:], z[:], indexing='ij')
Nx, Ny, Nz = len(x), len(y), len(z)

# Time data
Nframe = data['time/Nframe'][...]
dt = data['time/dt'][...]

# Calculate initial quantities
initial_wfn = [psiP2[:, :, :, 0], psiP1[:, :, :, 0], psi0[:, :, :, 0], psiM1[:, :, :, 0], psiM2[:, :, :, 0]]

# Total density
n = diag.calc_density(initial_wfn)

# Calculate normalised wavefunction
Zeta = diag.normalise_wfn(initial_wfn)

# Calculate spin vectors
fx, fy, fz = diag.calc_spin_vectors(initial_wfn[0], initial_wfn[1], initial_wfn[2], initial_wfn[3], initial_wfn[4])
F = np.sqrt(abs(fx) ** 2 + abs(fy) ** 2 + fz ** 2)

# Calculate spin expectation
spin_expec = F / n
spin_expec[n < 1e-6] = 0

# Calculate spin-singlet terms
a20 = diag.calc_spin_singlet_duo(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

a30 = diag.calc_spin_singlet_trio(Zeta[0], Zeta[1], Zeta[2], Zeta[3], Zeta[4])

# Generate figure:
fig = plt.figure(figsize=(12, 10))
grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.4, share_all=True, cbar_location="right",
                 cbar_mode="each", cbar_size="7%", cbar_pad=0.05)

# Plot titles:
grid[0].set_title(r'$|<\vec{F}>|$')
grid[1].set_title(r'$|A_{20}|^2$')
grid[2].set_title(r'$|A_{30}|^2$')

# Plot axis labels:
grid[0].set_ylabel(r'$y/\ell$')
for axis in grid:
    axis.set_xlabel(r'$x/\ell$')

# Initial plot
z_index = Nz // 2 - 10
extent = X.min(), X.max(), Y.min(), Y.max()
spin_plot = grid[0].imshow(abs(spin_expec[:, :, z_index]), extent=extent, vmin=0, vmax=2, cmap='jet')
a20_plot = grid[1].imshow(abs(a20[:, :, z_index]) ** 2, extent=extent, vmin=0, vmax=0.2, cmap='jet')
a30_plot = grid[2].imshow(abs(a30[:, :, z_index]) ** 2, extent=extent, vmin=0, vmax=2, cmap='jet')

# Set colorbars
grid[0].cax.colorbar(spin_plot, ticks=[0, 1, 2])
grid[1].cax.colorbar(a20_plot, ticks=[0, 1 / 5])
grid[2].cax.colorbar(a30_plot, ticks=[0, 2])

plt.suptitle(r'$t = 0$', y=0.7)
anim = animation.FuncAnimation(fig, animate, frames=num_of_frames)
anim.save(f'./animations/{filename}_BN.mp4', writer=animation.FFMpegWriter(fps=30))
