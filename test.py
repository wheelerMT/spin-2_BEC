import numpy as np
import cupy as cp
import h5py


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
    U[1, 2] = -cp.sqrt(3/8) * cp.exp(-1j * alpha) * cp.sin(2 * beta)
    U[1, 3] = -cp.exp(-1j * (alpha - gamma)) * S ** 2 * (S ** 2 - 3 * C ** 2)
    U[1, 4] = - 2 * cp.exp(-1j * (alpha - 2 * gamma)) * C * S ** 3

    U[2, 0] = cp.sqrt(6) * cp.exp(-2j * gamma) * C ** 2 * S ** 2
    U[2, 1] = cp.sqrt(3/8) * cp.exp(-1j * gamma) * cp.sin(2 * beta)
    U[2, 2] = 1 / 4 * (1 + 3 * cp.cos(2 * beta))
    U[2, 3] = -cp.sqrt(3/8) * cp.exp(1j * gamma) * cp.sin(2 * beta)
    U[2, 4] = cp.sqrt(6) * cp.exp(2j * gamma) * C ** 2 * S ** 2

    U[3, 0] = 2 * cp.exp(2j * (alpha - 2 * gamma)) * C * S ** 3
    U[3, 1] = -cp.exp(1j * (alpha - gamma)) * S ** 2 * (S ** 2 - 3 * C ** 2)
    U[3, 2] = cp.sqrt(3/8) * cp.exp(1j * alpha) * cp.sin(2 * beta)
    U[3, 3] = cp.exp(1j * (alpha + gamma)) * C ** 2 * (C ** 2 - 3 * S ** 2)
    U[3, 4] = - 2 * cp.exp(1j * (alpha + 2 * gamma)) * C ** 3 * S

    U[4, 0] = cp.exp(2j * (alpha - gamma)) * S ** 4
    U[4, 1] = 2 * cp.exp(1j * (2 * alpha + gamma)) * C * S ** 3
    U[4, 2] = cp.sqrt(6) * cp.exp(2j * alpha) * C ** 2 * S ** 2
    U[4, 3] = 2 * cp.exp(2j * (2 * alpha + gamma)) * C ** 3 * S
    U[4, 4] = cp.exp(2j * (alpha + gamma)) * C ** 4

    new_Wfn = []
    for jj in range(len(Wfn)):
        new_Wfn.append(U[jj, 0] * Wfn[0] + U[jj, 1] * Wfn[1] + U[jj, 2] * Wfn[2] + U[jj, 3] * Wfn[3] + U[jj, 4] * Wfn[4])

    return new_Wfn


def nonlin_evo(psiP2, psiP1, psi0, psiM1, psiM2, c0, c2, c4, V, p, dt, spin_f):
    # Calculate densities:
    n = abs(psiP2) ** 2 + abs(psiP1) ** 2 + abs(psi0) ** 2 + abs(psiM1) ** 2 + abs(psiM2) ** 2
    A00 = 1 / cp.sqrt(5) * (psi0 ** 2 - 2 * psiP1 * psiM1 + 2 * psiP2 * psiM2)
    fz = 2 * (abs(psiP2) ** 2 - abs(psiM2) ** 2) + abs(psiP1) ** 2 - abs(psiM1) ** 2

    # Evolve spin-singlet term -c4*(n^2-|alpha|^2)
    S = cp.sqrt(n ** 2 - abs(A00) ** 2)
    S = np.nan_to_num(S)

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
    Qfactor = 1j * (-4/3 * S1 + 1 / 6 * S2)
    Q2factor = (-5/4 + 4/3 * C1 - 1/12 * C2)
    Q3factor = 1j * (1/3 * S1 - 1/6 * S2)
    Q4factor = (1/4 - 1/3 * C1 + 1/12 * C2)

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


def first_kinetic_evo(Psi, A, B, C):
    """ Calculates kinetic term and then Fourier
        transforms back in positional space. """

    Wfn = [cp.fft.ifftn(A * Psi[0]),
           cp.fft.ifftn(B * Psi[1]),
           cp.fft.ifftn(C * Psi[2]),
           cp.fft.ifftn(B * Psi[3]),
           cp.fft.ifftn(A * Psi[4])]

    return Wfn


def last_kinetic_evo(Psi, A, B, C):
    """ Fourier transforms wfn first, then
        calculates kinetic term, leaving it
        in k-space. """

    Wfn = [A * cp.fft.fftn(Psi[0]),
           B * cp.fft.fftn(Psi[1]),
           C * cp.fft.fftn(Psi[2]),
           B * cp.fft.fftn(Psi[3]),
           A * cp.fft.fftn(Psi[4])]

    return Wfn


def get_TF_density(c0, c2, V):
    """ Get Thomas-Fermi profile using interaction parameters
        for a spin-2 condensate."""
    g = c0 + 4 * c2
    Tf = (0.5 * (15 * g / (4 * np.pi)) ** 0.4 - V) / g
    Tf = np.where(Tf < 0, 0, Tf)

    return Tf


# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny, Nz = 64, 64, 64  # Number of grid points
Mx, My, Mz = Nx // 2, Ny // 2, Nz // 2
dx, dy, dz = 0.5, 0.5, 0.5  # Grid spacing
dkx, dky, dkz = np.pi / (Mx * dx), np.pi / (My * dy), np.pi / (Mz * dz)  # K-space spacing
len_x, len_y, len_z = Nx * dx, Ny * dy, Nz * dz  # Box length

# Generate 1-D spatial grids:
x = cp.arange(-Mx, Mx) * dx
y = cp.arange(-My, My) * dy
z = cp.arange(-Mz, Mz) * dz
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

# Generate 1-D k-space grids:
kx = cp.arange(-Mx, Mx) * dkx
ky = cp.arange(-My, My) * dky
kz = cp.arange(-Mz, Mz) * dkz
Kx, Ky, Kz = cp.meshgrid(kx, ky, kz, indexing='ij')
# Kx, Ky, Kz = cp.fft.fftshift(Kx), cp.fft.fftshift(Ky), cp.fft.fftshift(Kz)

# Controlled variables:
spin_f = 2  # Spin-2
omega_trap = 0.75
V = 0.5 * omega_trap ** 2 * (X ** 2 + Y ** 2 + Z ** 2)
p = 0  # Linear Zeeman
q = -0.05  # Quadratic Zeeman
c0 = 5000
c2 = 1000
c4 = -1000

# Time steps, number and wavefunction save variables
Nt = 1000
Nframe = 100  # Saves data every Nframe time steps
eta = 1e-3
dt = -1j * 1e-2  # Time step
t = 0.

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
phi = cp.arctan2(Y, X)  # Phase is azimuthal angle around the core

Tf = get_TF_density(c0, c2, V)

# Generate initial wavefunctions:
psiP2 = cp.sqrt(Tf / 2) * cp.ones((Nx, Ny, Nz))
psiP1 = cp.zeros((Nx, Ny, Nz))
psi0 = cp.zeros((Nx, Ny, Nz))
psiM1 = cp.zeros((Nx, Ny, Nz))
psiM2 = cp.sqrt(Tf / 2) * cp.exp(1j * phi)

Psi = [psiP2, psiP1, psi0, psiM1, psiM2]  # Full 5x1 wavefunction

# Spin rotation on wavefunction:
alpha_angle = 0
beta_angle = 0.1
gamma_angle = 0
Psi = rotation(Psi, alpha_angle, beta_angle, gamma_angle)
N = [dx * dy * cp.sum(cp.abs(wfn) ** 2) for wfn in Psi]  # Atom number of each component
theta_fix = [cp.angle(wfn) for wfn in Psi]
Psi = [cp.fft.fftn(wfn) for wfn in Psi]  # Transforming wfn to Fourier space

# Coefficients for kinetic evolution:
Ek = cp.fft.fftshift(0.5 * (Kx ** 2 + Ky ** 2 + Kz ** 2))
A = cp.exp(-1j * (Ek + 4 * q) * dt / 2)
B = cp.exp(-1j * (Ek + q) * dt / 2)
C = cp.exp(-1j * Ek * dt / 2)

# --------------------------------------------------------------------------------------------------------------------
# Imaginary time:
# --------------------------------------------------------------------------------------------------------------------
for i in range(Nt):
    # Kinetic evolution:
    Psi = first_kinetic_evo(Psi, A, B, C)

    # Non-linear evolution:
    Psi = nonlin_evo(Psi[0], Psi[1], Psi[2], Psi[3], Psi[4], c0, c2, c4, V, p, dt, spin_f)

    # Kinetic evolution:
    Psi = last_kinetic_evo(Psi, A, B, C)

    # Renormalise  atom number and fix phase:
    for ii in range(len(Psi)):
        Psi[ii] = cp.fft.fftn(cp.sqrt(N[ii]) * cp.fft.ifftn(Psi[ii]) / cp.sqrt(cp.sum(abs(cp.fft.ifftn(Psi[ii])) ** 2)))
        Psi[ii] = cp.fft.fftn(abs(cp.fft.ifftn(Psi[ii])) * cp.exp(1j * theta_fix[ii]))

    if i % 100 == 0:
        print('t = {}'.format(i * dt))

with h5py.File('test_data.hdf5', 'w') as file:
    file.create_dataset('wavefunction/psiP2', data=cp.asnumpy(cp.fft.ifftn(Psi[0])))
    file.create_dataset('wavefunction/psiP1', data=cp.asnumpy(cp.fft.ifftn(Psi[1])))
    file.create_dataset('wavefunction/psi0', data=cp.asnumpy(cp.fft.ifftn(Psi[2])))
    file.create_dataset('wavefunction/psiM1', data=cp.asnumpy(cp.fft.ifftn(Psi[3])))
    file.create_dataset('wavefunction/psiM2', data=cp.asnumpy(cp.fft.ifftn(Psi[4])))
