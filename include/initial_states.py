from spin2 import *
from abc import ABC, abstractmethod


class InitialStateFactory:
    """Factory for choosing the correct initial state. Sets the initial state
    based on the type of interpolating spinor we require.
    """

    @staticmethod
    def set_initial_state(initial_state: str, grid: Grid, wfn: Wavefunction, eta: cp.ndarray) -> None:
        """Sets the initial state for the wavefunction. Takes in a :obj:`str` to
        determine which interpolating spinor we wish to use.

        Parameters
        ----------
        initial_state : str
            'C-BN', 'C-FM', 'UN-BN' or 'FM-BN' - String determining which
            initial interpolating spinor to use.
        grid : Grid
            Grid object associated with the wavefunction.
        wfn : Wavefunction
            Wavefunction object.
        eta : ndarray
            Interpolating parameter used for setting the type of interpolating
            initial state.
        """

        if initial_state == 'C-BN':
            CyclicBNInitialState.generate_initial_state(grid, wfn, eta)
        elif initial_state == 'C-FM':
            CyclicBNInitialState.generate_initial_state(grid, wfn, eta)
        elif initial_state == 'UN-BN':
            CyclicBNInitialState.generate_initial_state(grid, wfn, eta)
        elif initial_state == 'FM-BN':
            CyclicBNInitialState.generate_initial_state(grid, wfn, eta)


# Initial States
class TrappedInitialState(ABC):
    """Abstract class for the initial state of the condensate.
    """

    @staticmethod
    def get_TF_density(grid: Grid, wfn: Wavefunction) -> cp.ndarray:
        """Gets the 3D Thomas-Fermi profile for the condensate.

        Parameters
        ----------
        grid : Grid
            The Grid object associated with the wavefunction.
        wfn : Wavefunction
            The wavefunction object.

        Returns
        -------
        ndarray
            The 3D Thomas-Fermi profile.
        """

        g = wfn.c0 + 4 * wfn.c2
        Rtf = (15 * g / (4 * cp.pi)) ** 0.2

        r2 = grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2
        Tf_dens = 15 / (8 * cp.pi * Rtf ** 2) * (1 - r2 / Rtf ** 2)
        Tf_dens = cp.where(Tf_dens < 0, 0, Tf_dens)

        return Tf_dens

    @staticmethod
    @abstractmethod
    def generate_initial_state(grid: Grid, wfn: Wavefunction, eta: cp.ndarray) -> None:
        """Generates the initial state for the wavefunction.

        Parameters
        ----------
        grid : Grid
            The Grid object associated with the wavefunction.
        wfn : Wavefunction
            The wavefunction object.
        eta : ndarray
            The helper parameter that defines what kind of ground state
            manifold we are in. Takes different forms depending on the
            interface.
        """
        pass


class CyclicBNInitialState(TrappedInitialState):
    """Cyclic to BN interpolating spinor."""
    @staticmethod
    def generate_initial_state(grid: Grid, wfn: Wavefunction, eta: cp.ndarray):
        Tf = super().get_TF_density(grid, wfn)  # Get Thomas-Fermi profile

        psiP2 = cp.sqrt(Tf) * cp.sqrt((1 + eta ** 2)) / 2
        psiP1 = cp.zeros((grid.nx, grid.ny, grid.nz))
        psi0 = cp.sqrt(Tf) * 1j * cp.sqrt((1 - eta ** 2) / 2)
        psiM1 = cp.zeros((grid.nx, grid.ny, grid.nz))
        psiM2 = cp.sqrt(Tf) * cp.sqrt((1 + eta ** 2)) / 2

        wfn.set_initial_state(psiP2, psiP1, psi0, psiM1, psiM2)


class CyclicFMInitialState(TrappedInitialState):
    """Cyclic to FM interpolating spinor."""
    @staticmethod
    def generate_initial_state(grid: Grid, wfn: Wavefunction, eta: cp.ndarray):
        Tf = super().get_TF_density(grid, wfn)  # Get Thomas-Fermi profile

        psiP2 = cp.sqrt(Tf) * 1 / cp.sqrt(3) * cp.sqrt((1 + eta))
        psiP1 = cp.zeros((grid.nx, grid.ny, grid.nz))
        psi0 = cp.zeros((grid.nx, grid.ny, grid.nz))
        psiM1 = cp.sqrt(Tf) * 1 / cp.sqrt(3) * cp.sqrt((2 - eta))
        psiM2 = cp.zeros((grid.nx, grid.ny, grid.nz))

        wfn.set_initial_state(psiP2, psiP1, psi0, psiM1, psiM2)


class UNBNInitialState(TrappedInitialState):
    """UN to BN interpolating spinor."""
    @staticmethod
    def generate_initial_state(grid: Grid, wfn: Wavefunction, eta: cp.ndarray):
        Tf = super().get_TF_density(grid, wfn)  # Get Thomas-Fermi profile

        psiP2 = cp.sqrt(Tf) * cp.sqrt((1 - eta ** 2) / 2)
        psiP1 = cp.zeros((grid.nx, grid.ny, grid.nz))
        psi0 = cp.sqrt(Tf) * eta
        psiM1 = cp.zeros((grid.nx, grid.ny, grid.nz))
        psiM2 = cp.sqrt(Tf) * cp.sqrt((1 - eta ** 2) / 2)

        wfn.set_initial_state(psiP2, psiP1, psi0, psiM1, psiM2)


class FMBNInitialState(TrappedInitialState):
    """FM to BN interpolating spinor."""
    @staticmethod
    def generate_initial_state(grid: Grid, wfn: Wavefunction, eta: cp.ndarray):
        Tf = super().get_TF_density(grid, wfn)  # Get Thomas-Fermi profile

        psiP2 = cp.sqrt(Tf) * cp.sqrt((1 + eta)) / 2
        psiP1 = cp.zeros((grid.nx, grid.ny, grid.nz))
        psi0 = cp.zeros((grid.nx, grid.ny, grid.nz))
        psiM1 = cp.zeros((grid.nx, grid.ny, grid.nz))
        psiM2 = cp.sqrt(Tf) * cp.sqrt((1 - eta)) / 2

        wfn.set_initial_state(psiP2, psiP1, psi0, psiM1, psiM2)
