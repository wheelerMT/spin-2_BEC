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
        num_frames = Wfn[0].shape[-1]  # Number of frames of data

        # Calculate spin vectors:
        fp = np.empty((Nx, Ny, Nz, num_frames))
        fz = np.empty((Nx, Ny, Nz, num_frames))

        for i in num_frames:
            # Pull out wfn's frame by frame
            psiP2, psiP1, psi0, psiM1, psiM2 = Wfn[0][:, :, :, i], Wfn[1][:, :, :, i], Wfn[2][:, :, :, i], \
                                               Wfn[3][:, :, :, i], Wfn[4][:, :, :, i]
            fp[:, :, :, i] = ne.evaluate("sqrt(6) * (psiP1 * conj(psi0) + psi0 * conj(psiM1)) "
                                         "+ 2 * (psiM1 * conj(psiM2) + psiP2 * conj(psiP1))")
            fz[:, :, :, i] = ne.evaluate("2 * (abs(psiP2).real ** 2 - abs(psiM2).real ** 2) "
                                         "+ abs(psiP1).real ** 2 - abs(psiM1).real ** 2")

        return fp, fz

    else:  # Single frame of data
        psiP2, psiP1, psi0, psiM1, psiM2 = Wfn[0], Wfn[1], Wfn[2], Wfn[3], Wfn[4]
        fp = ne.evaluate("sqrt(6) * (psiP1 * conj(psi0) + psi0 * conj(psiM1)) "
                         "+ 2 * (psiM1 * conj(psiM2) + psiP2 * conj(psiP1))")
        fz = ne.evaluate("2 * (abs(psiP2).real ** 2 - abs(psiM2).real ** 2) "
                         "+ abs(psiP1).real ** 2 - abs(psiM1).real ** 2")

        return fp, fz


def calc_a20(psiP2, psiP1, psi0, psiM1, psiM2):
    """
    :param psiP2: psi_+2 component
    :param psiP1: psi_+1 component
    :param psi0: psi_0 component
    :param psiM1: psi_-1 component
    :param psiM2: psi_-2 component
    :return: a20, spin-singlet pair
    """

    # Total density
    n = abs(psiP2) ** 2 + abs(psiP1) ** 2 + abs(psi0) ** 2 + abs(psiM1) ** 2 + abs(psiM2) ** 2

    # Return a20
    return 1 / (np.sqrt(5) * n) * (2 * psiP2 * psiM2 - 2 * psiP1 * psiM1 + psi0 ** 2)


def calc_a30(psiP2, psiP1, psi0, psiM1, psiM2):
    """
    :param psiP2: psi_+2 component
    :param psiP1: psi_+1 component
    :param psi0: psi_0 component
    :param psiM1: psi_-1 component
    :param psiM2: psi_-2 component
    :return: a30, spin-singlet trio
    """

    # Total density
    n = abs(psiP2) ** 2 + abs(psiP1) ** 2 + abs(psi0) ** 2 + abs(psiM1) ** 2 + abs(psiM2) ** 2

    # Return a30
    return 1 / (n * np.sqrt(n)) * (3 * np.sqrt(6) / 2 * (psiP1 ** 2 * psiM2 + psiM1 ** 2 * psiP2) +
                                   psi0 * (psi0 ** 2 - 3 * psiP1 * psiM1 - 6 * psiP2 * psiM2))
