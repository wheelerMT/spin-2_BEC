import numexpr as ne

"""File containing functions for calculating certain diagnostics"""


def calc_spin_vectors(psiP2, psiP1, psi0, psiM1, psiM2):
    """
    :param psiP2: psi_+2 component
    :param psiP1: psi_+1 component
    :param psi0: psi_0 component
    :param psiM1: psi_-1 component
    :param psiM2: psi_-2 component
    :return: fp, fz: perpendicular and longitudinal spin vectors
    """

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
    return ne.evaluate("1 / (sqrt(5) * n) * (2 * psiP2 * psiM2 - 2 * psiP1 * psiM1 + psi0 ** 2)")


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
    return ne.evaluate("1 / (n * sqrt(n)) * (3 * np.sqrt(6) / 2 * (psiP1 ** 2 * psiM2 + psiM1 ** 2 * psiP2) "
                       "+ psi0 * (psi0 ** 2 - 3 * psiP1 * psiM1 - 6 * psiP2 * psiM2))")
