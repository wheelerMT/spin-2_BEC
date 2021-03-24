import numpy as np
import numexpr as ne

"""File containing functions for calculating certain diagnostics"""


def calc_spin_vectors(Wfn):
    """

    :param Wfn: 5x1 wavefunction
    :return: F_x+iF_y, Fz spin vectors
    """

    # Num of grid points in each dimension
    Nx, Ny, Nz = Wfn[0].shape[0], Wfn[0].shape[1], Wfn[0].shape[2]

    # Need to handle cases of differing dimensionality differently
    if Wfn[0].ndim == 4:  # Multiple frames of data
        numFrames = Wfn[0].shape[-1]  # Number of frames of data

        # Calculate spin vectors:
        fp = np.empty((Nx, Ny, Nz, numFrames))
        fz = np.empty((Nx, Ny, Nz, numFrames))

        for i in numFrames:
            # Pull out wfn's frame by frame
            psiP2, psiP1, psi0, psiM1, psiM2 = Wfn[0][:, :, :, i], Wfn[1][:, :, :, i], Wfn[2][:, :, :, i], \
                                               Wfn[3][:, :, :, i], Wfn[4][:, :, :, i]
            fp[:, :, :, i] = ne.evaluate("sqrt(6) * (psiP1 * conj(psi0) + psi0 * conj(psiM1)) "
                                         "+ 2 * (psiM1 * conj(psiM2) + psiP2 * conj(psiP1))")
            fz[:, :, :, i] = ne.evaluate("2 * (abs(psiP2).real ** 2 - abs(psiM2).real ** 2) "
                                         "+ abs(psiP1).real ** 2 - abs(psiM1).real ** 2")

        return fp, fz

    else:   # Single frame of data
        psiP2, psiP1, psi0, psiM1, psiM2 = Wfn[0], Wfn[1], Wfn[2], Wfn[3], Wfn[4]
        fp = ne.evaluate("sqrt(6) * (psiP1 * conj(psi0) + psi0 * conj(psiM1)) "
                         "+ 2 * (psiM1 * conj(psiM2) + psiP2 * conj(psiP1))")
        fz = ne.evaluate("2 * (abs(psiP2).real ** 2 - abs(psiM2).real ** 2) "
                         "+ abs(psiP1).real ** 2 - abs(psiM1).real ** 2")

        return fp, fz

