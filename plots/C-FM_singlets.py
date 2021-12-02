import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

Nx = 1000
eta_C_FM = np.zeros(Nx)
eta_C_FM[:Nx // 3] = -1
eta_C_FM[Nx // 3:2 * Nx // 3] = 0
eta_C_FM[2 * Nx // 3:] = 2

for i in range(100):
    eta_C_FM[Nx // 3 + i] = -1 + i / 100
    eta_C_FM[2 * Nx // 3 + i] = 0 + i / 50

mod_a30_C_FM = 0.5 * (2 - eta_C_FM) ** 2 * (1 + eta_C_FM)
plt.plot(eta_C_FM, mod_a30_C_FM, 'ko')
plt.show()
