"""Main file containing all necessary classes to simulate a spin- BEC."""

import cupy as cp
from typing import Iterator
import numpy as np


class Grid:
    """Generates a grid object for use with the wavefunction. Contains the properties
    of the grid as well as the 2D meshgrids themselves.

    Attributes
    ----------
    nx : int
        Number of x grid points.
    ny : int
        Number of y grid points.
    dx : float
        x grid spacing.
    dy : float
        y grid spacing.
    len_x : float
        Length of grid in x-direction.
    len_y : float
        Length of grid in y-direction.
    X : ndarray
        2D :obj:`ndarray` of X meshgrid.
    Y : ndarray
        2D :obj:`ndarray` of Y meshgrid.
    squared : ndarray
        2D :obj:`ndarray` storing result of X ** 2 + Y ** 2.
    """

    def __init__(self, nx: int, ny: int, dx: float = 1., dy: float = 1.):
        """Instantiate a Grid object.
        Automatically generates meshgrids using parameters provided.

        Parameters
        ----------
        nx : int
            Number of x grid points.
        ny : int
            Number of y grid points.
        dx : float
            Grid spacing for x grid, defaults to 1.
        dy : float
            Grid spacing for y grid, defaults to 1.
        """

        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.len_x, self.len_y = nx * dx, ny * dy

        # Generate 2D meshgrids:
        self.X, self.Y = cp.meshgrid(cp.arange(-nx // 2, nx // 2) * dx, cp.arange(-ny // 2, ny // 2) * dy)

        self.squared = self.X ** 2 + self.Y ** 2

    def fftshift(self):
        """Performs FFT shift on X & Y meshgrids.
        """

        self.X = cp.fft.fftshift(self.X)
        self.Y = cp.fft.fftshift(self.Y)


class Phase:
    """Generates the phase profile object for a distribution of vortices.
    Note: currently only random distribution is supported.

    Attributes
    ----------
    phase : ndarray
        2D array of the phase profile.
    """

    def __init__(self, nvort: int, thresh: float, grid: Grid, vortex_distribution: str):
        """Instantiate a phase object.
        Generates the phase grid with 2pi windings about each vortex.

        Parameters
        ----------
        nvort : int
            Total number of vortices initially in the system.
        thresh : float
            Threshold distance between any two vortices.
        grid : Grid
            The grid object associated with the wavefunction.
        vortex_distribution : str
            'random' - Defines type of vortex distribution.
        """

        self.phase = None

        if vortex_distribution == 'random':
            # If random is chosen, generate random positions then imprint
            initial_pos = self._generate_random_pos(nvort, thresh, grid)
            self._imprint_phase(nvort, grid, initial_pos)

    @staticmethod
    def _generate_random_pos(nvort: int, thresh: float, grid: Grid) \
            -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generates random positions using a uniform distribution.

        Parameters
        ----------
        nvort : int
            Total number of vortices initially in the system.
        thresh : float
            Threshold distance between any two vortices.
        grid : Grid
            The grid object associated with the wavefunction.

        Returns
        -------
        Iterator[tuple[np.ndarray, np.ndarray]]
            Iterator of vortex positions.

        """

        accepted_pos = []
        iterations = 0
        while len(accepted_pos) < nvort:
            within_range = True
            while within_range:
                iterations += 1
                triggered = False

                # Generate a position
                pos = cp.random.uniform(-grid.len_x // 2, grid.len_x // 2), \
                      cp.random.uniform(-grid.len_y // 2, grid.len_y // 2)

                # Check if position is too close to any other position
                for accepted_position in accepted_pos:
                    if abs(pos[0] - accepted_position[0]) < thresh:
                        if abs(pos[1] - accepted_position[1]) < thresh:
                            triggered = True
                            break

                # If position isn't close to others then add it to accepted positions
                if not triggered:
                    accepted_pos.append(pos)
                    within_range = False

            # Prints out current progress every 500 iterations
            if len(accepted_pos) % 500 == 0:
                print('Found {} positions...'.format(len(accepted_pos)))

        print('Found {} positions in {} iterations.'.format(len(accepted_pos), iterations))
        return iter(accepted_pos)  # Set accepted positions to member

    def _imprint_phase(self, nvort: int, grid: Grid, pos: Iterator[tuple[np.ndarray, np.ndarray]]):
        """Imprints 2pi windings in the phase using the positions provided
        then updates phase attribute with the result.

        Parameters
        ----------
        nvort : int
            Total number of vortices initially in the system.
        grid : Grid
            The grid object associated with the wavefunction.
        pos : Iterator[tuple[np.ndarray, np.ndarray]]
            Iterator of vortex positions.

        """

        # Initialise phase:
        theta_tot = cp.empty((grid.nx, grid.ny))

        # Scale pts:
        x_tilde = 2 * cp.pi * ((grid.X - grid.X.min()) / grid.len_x)
        y_tilde = 2 * cp.pi * ((grid.Y - grid.Y.min()) / grid.len_y)

        # Construct phase for this postion:
        for _ in range(nvort):
            theta_k = cp.zeros((grid.nx, grid.ny))

            try:
                x_m, y_m = next(pos)
                x_p, y_p = next(pos)
            except StopIteration:
                break

            # Scaling vortex positions:
            x_m_tilde = 2 * cp.pi * ((x_m - grid.X.min()) / grid.len_x)
            y_m_tilde = 2 * cp.pi * ((y_m - grid.Y.min()) / grid.len_y)
            x_p_tilde = 2 * cp.pi * ((x_p - grid.X.min()) / grid.len_x)
            y_p_tilde = 2 * cp.pi * ((y_p - grid.Y.min()) / grid.len_y)

            # Aux variables
            Y_minus = y_tilde - y_m_tilde
            X_minus = x_tilde - x_m_tilde
            Y_plus = y_tilde - y_p_tilde
            X_plus = x_tilde - x_p_tilde

            heav_xp = cp.asarray(np.heaviside(cp.asnumpy(X_plus), 1.))
            heav_xm = cp.asarray(np.heaviside(cp.asnumpy(X_minus), 1.))

            for nn in cp.arange(-5, 6):
                theta_k += cp.arctan(cp.tanh((Y_minus + 2 * cp.pi * nn) / 2) * cp.tan((X_minus - cp.pi) / 2)) \
                           - cp.arctan(cp.tanh((Y_plus + 2 * cp.pi * nn) / 2) * cp.tan((X_plus - cp.pi) / 2)) \
                           + cp.pi * (heav_xp - heav_xm)

            theta_k -= y_tilde * (x_p_tilde - x_m_tilde) / (2 * cp.pi)
            theta_tot += theta_k

        self.phase = theta_tot
