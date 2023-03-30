import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import diagnostics as diag

plt.rcParams["text.usetex"] = True
plt.rc("text.latex", preamble=r"\usepackage{txfonts}")
plt.rcParams.update({"font.size": 26})

path = Path("data/3D/frames")
filename = "6f_C-FM=2_coreless.hdf5"

# Load data
data = h5py.File(path / filename, "r")
x, y, z = data["grid/x"], data["grid/y"], data["grid/z"]
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
Nx, Ny, Nz = len(x), len(y), len(z)
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
psip2 = data["wavefunction/psiP2"][:, :, :, 2]
psip1 = data["wavefunction/psiP1"][:, :, :, 2]
psi0 = data["wavefunction/psi0"][:, :, :, 2]
psim1 = data["wavefunction/psiM1"][:, :, :, 2]
psim2 = data["wavefunction/psiM2"][:, :, :, 2]

zetap2, zetap1, zeta0, zetam1, zetam2 = diag.normalise_wfn(
    [psip2, psip1, psi0, psim1, psim2]
)

# Tophat for smoothing plots
sigma = 0.5
c0 = 1.32e4
c2 = 146
g = c0 + 4 * c2
rtf = (15 * g / (4 * np.pi)) ** 0.2
tophat = 0.5 * (1 - np.tanh(sigma * (X**2 + Y**2 + Z**2 - rtf**2)))

fx, fy, fz = diag.calc_spin_vectors(zetap2, zetap1, zeta0, zetam1, zetam2)
spin_expec = tophat * np.sqrt(abs(fx) ** 2 + abs(fy) ** 2 + abs(fz) ** 2)

# Construct array of diagonal
spin_expec_diag = np.empty((Nx, Nz))
for i in range(Nz):
    spin_expec_diag[:, i] = np.diag(spin_expec[:, :, i])

# Construct plot at constant y
fig, ax = plt.subplots(1, figsize=(4.2, 3.54))
ax.set_xlim(-X.max() + 2, X.max() - 2)
ax.set_ylim(-X.max() + 2, X.max() - 2)
ax.set_xlabel(r"$(x + y)/\sqrt{2}\ell$")
ax.set_ylabel(r"$z/\ell$")

y_index = Ny // 2
extent = (
    (X.min() + Y.min()) / np.sqrt(2),
    (X.max() + Y.max()) / np.sqrt(2),
    Z.min(),
    Z.max(),
)
consty_plot = ax.imshow(
    spin_expec_diag.T,
    origin="lower",
    extent=extent,
    vmin=0,
    vmax=2,
    cmap="jet",
    interpolation="gaussian",
)
ax.plot([-7, 7], [0, 0], "w--", linewidth=3)
constz_cbar = plt.colorbar(consty_plot, ax=ax, pad=0.01)
constz_cbar.set_ticks([0, 1, 2])
constz_cbar.set_ticklabels([r"$0$", r"$1$", r"$2$"])
constz_cbar.set_label(r"$|\langle \hat{\mathbf{F}} \rangle|$", labelpad=12)
plt.savefig(
    "../plots/spin-2/C-FM=2_third-SQV_longitudinal_spin_mag.pdf",
    bbox_inches="tight",
    dpi=200,
)
plt.show()
plt.show()
