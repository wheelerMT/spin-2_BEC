import numpy as np
from numba import njit

"""File containing functions for calculating certain diagnostics accelerated using numba"""


def calc_density(Wfn):
    """
    :param Wfn: List, each element is a wavefunction component starting with psi_+2
    :return: n: total atomic density
    """

    return sum(abs(wfn) ** 2 for wfn in Wfn)


def normalise_wfn(Wfn):
    """
    :param Wfn: List, each element is a wavefunction component starting with psi_+2
    :return: Normalised 5 x 1 wavefunction
    """

    # Density
    n = calc_density(Wfn)

    # Normalise wavefunction
    Zeta = [wfn / np.sqrt(n) for wfn in Wfn]

    # Correct division by small numbers outside of trap
    for zeta in Zeta:
        zeta[n < 1e-6] = 0

    return Zeta


@njit
def calc_spin_vectors(psiP2, psiP1, psi0, psiM1, psiM2):
    """
    :param psiP2: psi_+2 component
    :param psiP1: psi_+1 component
    :param psi0: psi_0 component
    :param psiM1: psi_-1 component
    :param psiM2: psi_-2 component
    :return: fp, fz: perpendicular and longitudinal spin vectors
    """

    fp = np.sqrt(6) * (psiP1 * np.conj(psi0) + psi0 * np.conj(psiM1)) + \
         2 * (psiM1 * np.conj(psiM2) + psiP2 * np.conj(psiP1))
    fz = 2 * (np.abs(psiP2) ** 2 - np.abs(psiM2) ** 2) + np.abs(psiP1) ** 2 - np.abs(psiM1) ** 2

    return fp, fz


@njit
def calc_spin_singlet_duo(zetaP2, zetaP1, zeta0, zetaM1, zetaM2):
    """
    :param zetaP2: normalised psi_+2 component
    :param zetaP1: normalised psi_+1 component
    :param zeta0: normalised psi_0 component
    :param zetaM1: normalised psi_-1 component
    :param zetaM2: normalised psi_-2 component
    :return: a20, spin-singlet duo
    """

    # Return a20
    return 1 / (np.sqrt(5)) * (2 * zetaP2 * zetaM2 - 2 * zetaP1 * zetaM1 + zeta0 ** 2)


@njit
def calc_spin_singlet_trio(zetaP2, zetaP1, zeta0, zetaM1, zetaM2):
    """
    :param zetaP2: normalised psi_+2 component
    :param zetaP1: normalised psi_+1 component
    :param zeta0: normalised psi_0 component
    :param zetaM1: normalised psi_-1 component
    :param zetaM2: normalised psi_-2 component
    :return: a30, spin-singlet trio
    """

    # Return a30
    return (3 * np.sqrt(6) / 2 * (zetaP1 ** 2 * zetaM2 + zetaM1 ** 2 * zetaP2)
            + zeta0 * (zeta0 ** 2 - 3 * zetaP1 * zetaM1 - 6 * zetaP2 * zetaM2))
