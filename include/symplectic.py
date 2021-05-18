import cupy as cp


def calc_Qpsi(fz, fp, Wfn):
    Qpsi = [2 * fz * Wfn[0] + fp * Wfn[1],
            cp.conj(fp) * Wfn[0] + fz * Wfn[1] + cp.sqrt(3 / 2) * fp * Wfn[2],
            cp.sqrt(3 / 2) * (cp.conj(fp) * Wfn[1] + fp * Wfn[3]),
            cp.sqrt(3 / 2) * cp.conj(fp) * Wfn[2] - fz * Wfn[3] + fp * Wfn[4],
            cp.conj(fp) * Wfn[3] - 2 * fz * Wfn[4]]

    return Qpsi


def rotation(Wfn, alpha, beta, gamma):
    U = cp.empty((5, 5))
    C = cp.cos(beta / 2)
    S = cp.sin(beta / 2)

    U[0, 0] = cp.exp(-2j * (alpha + gamma)) * C ** 4
    U[0, 1] = -2 * cp.exp(-1j * (2 * alpha + gamma)) * C ** 3 * S
    U[0, 2] = cp.sqrt(6) * cp.exp(-2j * alpha) * C ** 2 * S ** 2
    U[0, 3] = - 2 * cp.exp(-1j * (2 * alpha - gamma)) * C * S ** 3
    U[0, 4] = cp.exp(-2j * (alpha - gamma)) * S ** 4

    U[1, 0] = 2 * cp.exp(-1j * (alpha + 2 * gamma)) * C ** 3 * S
    U[1, 1] = cp.exp(-1j * (alpha + gamma)) * C ** 2 * (C ** 2 - 3 * S ** 2)
    U[1, 2] = -cp.sqrt(3 / 8) * cp.exp(-1j * alpha) * cp.sin(2 * beta)
    U[1, 3] = -cp.exp(-1j * (alpha - gamma)) * S ** 2 * (S ** 2 - 3 * C ** 2)
    U[1, 4] = - 2 * cp.exp(-1j * (alpha - 2 * gamma)) * C * S ** 3

    U[2, 0] = cp.sqrt(6) * cp.exp(-2j * gamma) * C ** 2 * S ** 2
    U[2, 1] = cp.sqrt(3 / 8) * cp.exp(-1j * gamma) * cp.sin(2 * beta)
    U[2, 2] = 1 / 4 * (1 + 3 * cp.cos(2 * beta))
    U[2, 3] = -cp.sqrt(3 / 8) * cp.exp(1j * gamma) * cp.sin(2 * beta)
    U[2, 4] = cp.sqrt(6) * cp.exp(2j * gamma) * C ** 2 * S ** 2

    U[3, 0] = 2 * cp.exp(2j * (alpha - 2 * gamma)) * C * S ** 3
    U[3, 1] = -cp.exp(1j * (alpha - gamma)) * S ** 2 * (S ** 2 - 3 * C ** 2)
    U[3, 2] = cp.sqrt(3 / 8) * cp.exp(1j * alpha) * cp.sin(2 * beta)
    U[3, 3] = cp.exp(1j * (alpha + gamma)) * C ** 2 * (C ** 2 - 3 * S ** 2)
    U[3, 4] = - 2 * cp.exp(1j * (alpha + 2 * gamma)) * C ** 3 * S

    U[4, 0] = cp.exp(2j * (alpha - gamma)) * S ** 4
    U[4, 1] = 2 * cp.exp(1j * (2 * alpha + gamma)) * C * S ** 3
    U[4, 2] = cp.sqrt(6) * cp.exp(2j * alpha) * C ** 2 * S ** 2
    U[4, 3] = 2 * cp.exp(2j * (2 * alpha + gamma)) * C ** 3 * S
    U[4, 4] = cp.exp(2j * (alpha + gamma)) * C ** 4

    new_Wfn = []
    for jj in range(len(Wfn)):
        new_Wfn.append(
            U[jj, 0] * Wfn[0] + U[jj, 1] * Wfn[1] + U[jj, 2] * Wfn[2] + U[jj, 3] * Wfn[3] + U[jj, 4] * Wfn[4])

    return new_Wfn


def nonlin_evo(psiP2, psiP1, psi0, psiM1, psiM2, c0, c2, c4, V, p, dt, spin_f):
    # Calculate densities:
    n = abs(psiP2) ** 2 + abs(psiP1) ** 2 + abs(psi0) ** 2 + abs(psiM1) ** 2 + abs(psiM2) ** 2
    A00 = 1 / cp.sqrt(5) * (psi0 ** 2 - 2 * psiP1 * psiM1 + 2 * psiP2 * psiM2)
    fz = 2 * (abs(psiP2) ** 2 - abs(psiM2) ** 2) + abs(psiP1) ** 2 - abs(psiM1) ** 2

    # Evolve spin-singlet term -c4*(n^2-|alpha|^2)
    S = cp.sqrt(n ** 2 - abs(A00) ** 2)
    S = cp.nan_to_num(S)

    cosT = cp.cos(c4 * S * dt)
    sinT = cp.sin(c4 * S * dt) / S
    sinT[S == 0] = 0  # Corrects division by 0

    Wfn = [psiP2 * cosT + 1j * (n * psiP2 - A00 * cp.conj(psiM2)) * sinT,
           psiP1 * cosT + 1j * (n * psiP1 + A00 * cp.conj(psiM1)) * sinT,
           psi0 * cosT + 1j * (n * psi0 - A00 * cp.conj(psi0)) * sinT,
           psiM1 * cosT + 1j * (n * psiM1 + A00 * cp.conj(psiP1)) * sinT,
           psiM2 * cosT + 1j * (n * psiM2 - A00 * cp.conj(psiP2)) * sinT]

    # Calculate spin vectors
    fp = cp.sqrt(6) * (Wfn[1] * cp.conj(Wfn[2]) + Wfn[2] * cp.conj(Wfn[3])) + 2 * (Wfn[3] * cp.conj(Wfn[4]) +
                                                                                   Wfn[0] * cp.conj(Wfn[1]))
    F = cp.sqrt(fz ** 2 + abs(fp) ** 2)

    # Calculate cos, sin and Qfactor terms:
    C1, S1 = cp.cos(c2 * F * dt), cp.sin(c2 * F * dt)
    C2, S2 = cp.cos(2 * c2 * F * dt), cp.sin(2 * c2 * F * dt)
    Qfactor = 1j * (-4 / 3 * S1 + 1 / 6 * S2)
    Q2factor = (-5 / 4 + 4 / 3 * C1 - 1 / 12 * C2)
    Q3factor = 1j * (1 / 3 * S1 - 1 / 6 * S2)
    Q4factor = (1 / 4 - 1 / 3 * C1 + 1 / 12 * C2)

    fzQ = fz / F
    fpQ = fp / F

    Qpsi = calc_Qpsi(fzQ, fpQ, Wfn)
    Q2psi = calc_Qpsi(fzQ, fpQ, Qpsi)
    Q3psi = calc_Qpsi(fzQ, fpQ, Q2psi)
    Q4psi = calc_Qpsi(fzQ, fpQ, Q3psi)

    # Evolve spin term c2 * F^2
    for ii in range(len(Wfn)):
        Wfn[ii] += Qfactor * Qpsi[ii] + Q2factor * Q2psi[ii] + Q3factor * Q3psi[ii] + Q4factor * Q4psi[ii]

    # Evolve (c0+c4)*n^2 + (V + pm)*n:
    for ii in range(len(Wfn)):
        mF = spin_f - ii
        Wfn[ii] *= cp.exp(-1j * dt * ((c0 + c4) * n + V + p * mF))

    return Wfn


def first_kinetic_evo(Wfn, A, B, C):
    """ Calculates kinetic term and then Fourier
        transforms back in positional space. """

    Wfn = [cp.fft.ifftn(A * Wfn[0]),
           cp.fft.ifftn(B * Wfn[1]),
           cp.fft.ifftn(C * Wfn[2]),
           cp.fft.ifftn(B * Wfn[3]),
           cp.fft.ifftn(A * Wfn[4])]

    return Wfn


def last_kinetic_evo(Wfn, A, B, C):
    """ Fourier transforms wfn first, then
        calculates kinetic term, leaving it
        in k-space. """

    Wfn = [A * cp.fft.fftn(Wfn[0]),
           B * cp.fft.fftn(Wfn[1]),
           C * cp.fft.fftn(Wfn[2]),
           B * cp.fft.fftn(Wfn[3]),
           A * cp.fft.fftn(Wfn[4])]

    return Wfn


def first_kinetic_rot_evo_3d(Wfn, X, Y, Kx, Ky, Kz, omega, spin_f, q, dt):
    for ii in range(len(Wfn)):
        mF = spin_f - ii

        # IFFT over y, so we are in x and z directions
        Wfn[ii] = cp.fft.ifft(Wfn[ii], axis=1)

        # Compute first coefficient rotation
        Wfn[ii] *= cp.exp(-1j * dt * (2 * Kx ** 2 + Kz ** 2 + 4 * omega * Y * Kx + mF ** 2 * q / 2) / 8)

        # IFFT over x, FFT over y, so we are in y and z directions
        Wfn[ii] = cp.fft.ifft(Wfn[ii], axis=0)
        Wfn[ii] = cp.fft.fft(Wfn[ii], axis=1)

        # Compute second coefficient rotation:
        Wfn[ii] *= cp.exp(-1j * dt * (2 * Ky ** 2 + Kz ** 2 - 4 * omega * X * Ky + mF ** 2 * q / 2) / 8)

        # IFFT over y and z, so all axes are in position space
        Wfn[ii] = cp.fft.ifftn(Wfn[ii], axes=(1, 2))


def last_kinetic_rot_evo_3d(Wfn, X, Y, Kx, Ky, Kz, omega, spin_f, q, dt):
    for ii in range(len(Wfn)):
        mF = spin_f - ii

        # FFT over y and z, so we are in y and z directions
        Wfn[ii] = cp.fft.fftn(Wfn[ii], axes=(1, 2))

        # Compute second coefficient rotation
        Wfn[ii] *= cp.exp(-1j * dt * (2 * Ky ** 2 + Kz ** 2 - 4 * omega * X * Ky + mF ** 2 * q / 2) / 8)

        # IFFT over y and FFT over x, so we are in x and z directions
        Wfn[ii] = cp.fft.ifft(Wfn[ii], axis=1)
        Wfn[ii] = cp.fft.fft(Wfn[ii], axis=0)

        # Compute first coefficient rotation
        Wfn[ii] *= cp.exp(-1j * dt * (2 * Kx ** 2 + Kz ** 2 + 4 * omega * Y * Kx + mF ** 2 * q / 2) / 8)

        # FFT over y, so all axes are in Fourier space
        Wfn[ii] = cp.fft.fft(Wfn[ii], axis=1)


def first_kinetic_rot_evo_2d(Wfn, X, Y, Kx, Ky, omega, spin_f, q, dt):
    for ii in range(len(Wfn)):
        mF = spin_f - ii

        # IFFT over y, so we are in x direction
        Wfn[ii] = cp.fft.ifft(Wfn[ii], axis=1)

        # Compute first coefficient rotation
        Wfn[ii] *= cp.exp(-1j * dt * (2 * Kx ** 2 + 4 * omega * Y * Kx + mF ** 2 * q / 2) / 8)

        # IFFT over x, FFT over y, so we are in y direction
        Wfn[ii] = cp.fft.ifft(Wfn[ii], axis=0)
        Wfn[ii] = cp.fft.fft(Wfn[ii], axis=1)

        # Compute second coefficient rotation:
        Wfn[ii] *= cp.exp(-1j * dt * (2 * Ky ** 2 - 4 * omega * X * Ky + mF ** 2 * q / 2) / 8)

        # IFFT over y so all axes are in position space
        Wfn[ii] = cp.fft.ifft(Wfn[ii], axis=1)


def last_kinetic_rot_evo_2d(Wfn, X, Y, Kx, Ky, omega, spin_f, q, dt):
    for ii in range(len(Wfn)):
        mF = spin_f - ii

        # FFT over y, so we are in y and z directions
        Wfn[ii] = cp.fft.fft(Wfn[ii], axis=1)

        # Compute second coefficient rotation
        Wfn[ii] *= cp.exp(-1j * dt * (2 * Ky ** 2 - 4 * omega * X * Ky + mF ** 2 * q / 2) / 8)

        # IFFT over y and FFT over x, so we are in x and z directions
        Wfn[ii] = cp.fft.ifft(Wfn[ii], axis=1)
        Wfn[ii] = cp.fft.fft(Wfn[ii], axis=0)

        # Compute first coefficient rotation
        Wfn[ii] *= cp.exp(-1j * dt * (2 * Kx ** 2 + 4 * omega * Y * Kx + mF ** 2 * q / 2) / 8)

        # FFT over y, so all axes are in Fourier space
        Wfn[ii] = cp.fft.fft(Wfn[ii], axis=1)


def get_TF_density_3d(c0, c2, X, Y, Z, N):
    """ Get 3D Thomas-Fermi profile using interaction parameters
        for a spin-2 condensate."""
    g = c0 + 4 * c2
    Rtf = (15 * N * g / (4 * cp.pi)) ** 0.2

    r2 = X ** 2 + Y ** 2 + Z ** 2
    Tf_dens = 15 * N / (8 * cp.pi * Rtf ** 2) * (1 - r2 / Rtf ** 2)
    Tf_dens = cp.where(Tf_dens < 0, 0, Tf_dens)

    return Tf_dens


def get_TF_density_2d(c0, c2, X, Y, N):
    """ Get 2D Thomas-Fermi profile using interaction parameters
        for a spin-2 condensate."""
    g = c0 + 4 * c2
    Rtf = (8 * N * g / cp.pi) ** 0.25

    r2 = X ** 2 + Y ** 2
    Tf_dens = 4 * N / (cp.pi * Rtf ** 2) * (1 - r2 / Rtf ** 2)
    Tf_dens = cp.where(Tf_dens < 0, 0, Tf_dens)

    return Tf_dens
