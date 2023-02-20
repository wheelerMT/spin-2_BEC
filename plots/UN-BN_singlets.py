import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams.update({"font.size": 14})

matplotlib.use("TkAgg")

Nx = 1000
eta_UN_BN = np.linspace(-1, 1, Nx)
theta = np.linspace(0, 2 * np.pi, Nx)
Eta, Theta = np.meshgrid(eta_UN_BN, theta, indexing="ij")

mod_a20_UN_BN = 0.1 * ((1 - Eta**2) * np.cos(Theta) + Eta**2 + 1)
mod_a30_UN_BN = (
    (1 + Eta)
    / 4
    * (3 * (Eta**2 - 1) * np.cos(Theta) + Eta * (5 * Eta - 8) + 5)
)


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
for axis in ax:
    axis.set_box_aspect(1)

ax[0].set_xlabel(r"$\eta$")
ax[1].set_xlabel(r"$\eta$")
ax[0].set_ylabel(r"$\theta$")

ax[0].set_yticks([0, np.pi, 2 * np.pi])
ax[0].set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])

a20_plot = ax[0].contourf(Eta, Theta, mod_a20_UN_BN, levels=100, cmap="jet")
a30_plot = ax[1].contourf(Eta, Theta, mod_a30_UN_BN, levels=100, cmap="jet")

a20_cbar = fig.colorbar(
    a20_plot, ax=ax[0], orientation="vertical", fraction=0.047, pad=0.02
)
a20_cbar.set_ticks([0, 0.2])
a20_cbar.set_ticklabels(["0", "0.2"])


a30_cbar = fig.colorbar(
    a30_plot, ax=ax[1], orientation="vertical", fraction=0.047, pad=0.02
)
a30_cbar.set_ticks([0, 1, 2])
a30_cbar.set_ticklabels(["0", "1", "2"])

plt.savefig(
    "../plots/spin-2/a20-a30-varying.png", dpi=200, bbox_inches="tight"
)
plt.show()
