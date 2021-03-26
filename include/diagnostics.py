import numpy as np
from numba import njit

"""File containing functions for calculating certain diagnostics accellerated using numba"""


def calc_density(psiP2, psiP1, psi0, psiM1, psiM2):
    """
    :param psiP2: psi_+2 component
    :param psiP1: psi_+1 component
    :param psi0: psi_0 component
    :param psiM1: psi_-1 component
    :param psiM2: psi_-2 component
    :return: n: total atomic density
    """

    return abs(psiP2) ** 2 + abs(psiP1) ** 2 + abs(psi0) ** 2 + abs(psiM1) ** 2 + abs(psiM2) ** 2


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
    fz = 2 * (np.abs(psiP2).real ** 2 - np.abs(psiM2).real ** 2) + np.abs(psiP1).real ** 2 - np.abs(psiM1).real ** 2

    return fp, fz


def normalise_wfn(psiP2, psiP1, psi0, psiM1, psiM2):
    """
    :param psiP2: psi_+2 component
    :param psiP1: psi_+1 component
    :param psi0: psi_0 component
    :param psiM1: psi_-1 component
    :param psiM2: psi_-2 component
    :return: Normalised wavefunction components (same order as input)
    """

    # Density
    n = calc_density(psiP2, psiP1, psi0, psiM1, psiM2)

    return psiP2 / np.sqrt(n), psiP1 / np.sqrt(n), psi0 / np.sqrt(n), psiM1 / np.sqrt(n), psiM2 / np.sqrt(n)


@njit
def calc_spin_singlet_duo(psiP2, psiP1, psi0, psiM1, psiM2):
    """
    :param psiP2: psi_+2 component
    :param psiP1: psi_+1 component
    :param psi0: psi_0 component
    :param psiM1: psi_-1 component
    :param psiM2: psi_-2 component
    :return: a20, spin-singlet duo
    """

    # Total density
    n = np.abs(psiP2) ** 2 + np.abs(psiP1) ** 2 + np.abs(psi0) ** 2 + np.abs(psiM1) ** 2 + np.abs(psiM2) ** 2

    # Return a20
    return 1 / (np.sqrt(5) * n) * (2 * psiP2 * psiM2 - 2 * psiP1 * psiM1 + psi0 ** 2)


@njit
def calc_spin_singlet_trio(psiP2, psiP1, psi0, psiM1, psiM2):
    """
    :param psiP2: psi_+2 component
    :param psiP1: psi_+1 component
    :param psi0: psi_0 component
    :param psiM1: psi_-1 component
    :param psiM2: psi_-2 component
    :return: a30, spin-singlet trio
    """

    # Total density
    n = np.abs(psiP2) ** 2 + np.abs(psiP1) ** 2 + np.abs(psi0) ** 2 + np.abs(psiM1) ** 2 + np.abs(psiM2) ** 2

    # Return a30
    return 1 / (n * np.sqrt(n)) * (3 * np.sqrt(6) / 2 * (psiP1 ** 2 * psiM2 + psiM1 ** 2 * psiP2)
                                   + psi0 * (psi0 ** 2 - 3 * psiP1 * psiM1 - 6 * psiP2 * psiM2))
