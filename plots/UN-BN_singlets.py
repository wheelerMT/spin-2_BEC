import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

Nx = 1000
eta_UN_BN = np.zeros(Nx)
eta_UN_BN[:Nx // 2] = 1
for i in range(100):
    eta_UN_BN[Nx // 2 - i] = 0 + i / 100
theta = np.linspace(0, np.pi, Nx)
Eta, Theta = np.meshgrid(eta_UN_BN, theta, indexing='ij')

mod_a30_UN_BN = Eta ** 2 * (Eta ** 4 - 6 * Eta ** 2 * (1 - Eta ** 2) * np.cos(2 * Theta)
                            + 9 * (1 - Eta ** 2) ** 2)
mod_a20_UN_BN = 0.2 * ((1 - Eta ** 2) ** 2 + 2 * Eta ** 2 * (1 - Eta ** 2) * np.cos(2 * Theta) + Eta ** 4)

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(3.54, 3.54), dpi=600)
a30_plot = ax[1].contourf(Eta, Theta, mod_a30_UN_BN, levels=200, cmap='jet')
ax[0].set_xlabel(r'$\eta$')
ax[0].set_yticks([0, np.pi / 2, np.pi])
ax[0].set_yticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$'])
ax[0].set_ylabel(r'$\theta$')
ax[0].set_title(r'$|a_{20}|^2$')
ax[0].set_aspect(1.0 / ax[0].get_data_ratio(), adjustable='box')
fig.colorbar(a30_plot, ax=ax[1], ticks=[0, 1, 2], fraction=0.046, pad=0.04)

a20_plot = ax[0].contourf(Eta, Theta, mod_a20_UN_BN, levels=200, cmap='jet')
ax[1].set_xlabel(r'$\eta$')
ax[1].set_title(r'$|a_{30}|^2$')
ax[1].set_aspect(1.0 / ax[1].get_data_ratio(), adjustable='box')
fig.colorbar(a20_plot, ax=ax[0], ticks=[0, 1 / 5], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('../../plots/spin-2/a20-a30-varying.png', bbox_inches='tight')
plt.show()
