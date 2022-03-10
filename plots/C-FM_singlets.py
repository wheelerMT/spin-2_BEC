import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

Nx = 1000
eta_C_FM = np.linspace(-1, 2, Nx)

mod_a30_C_FM = 0.5 * (2 - eta_C_FM) ** 2 * (1 + eta_C_FM)
plt.plot(eta_C_FM, mod_a30_C_FM, 'k')
plt.ylabel(r'$|A_{30}|^2$')
plt.xlabel(r'$\eta$')
plt.show()
