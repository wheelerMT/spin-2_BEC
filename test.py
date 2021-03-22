import numpy as np
import cupy as cp
import include.symplectic as sm
import h5py


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
Kx, Ky, Kz = cp.fft.fftshift(Kx), cp.fft.fftshift(Ky), cp.fft.fftshift(Kz)

# Controlled variables:
spin_f = 2  # Spin-2
omega_rot = 0.2
omega_trap = 0.5
V = 0.5 * omega_trap ** 2 * (X ** 2 + Y ** 2 + Z ** 2)
p = 0  # Linear Zeeman
q = -0.05  # Quadratic Zeeman
c0 = 5000
c2 = 1000
c4 = -1000

# Time steps, number and wavefunction save variables
Nt = 1000
Nframe = 100  # Saves data every Nframe time steps
dt = -1j * 1e-2  # Time step
t = 0.

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
phi = cp.arctan2(Y - 1, X - 1)  # Phase is azimuthal angle around the core

Tf = sm.get_TF_density(c0, c2, V)

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
Psi = sm.rotation(Psi, alpha_angle, beta_angle, gamma_angle)
N = [dx * dy * cp.sum(cp.abs(wfn) ** 2) for wfn in Psi]  # Atom number of each component
theta_fix = [cp.angle(wfn) for wfn in Psi]
Psi = [cp.fft.fftn(wfn) for wfn in Psi]  # Transforming wfn to Fourier space

# Coefficients for kinetic evolution:
Ek = 0.5 * (Kx ** 2 + Ky ** 2 + Kz ** 2)
A = cp.exp(-1j * (Ek + 4 * q) * dt / 2)
B = cp.exp(-1j * (Ek + q) * dt / 2)
C = cp.exp(-1j * Ek * dt / 2)

# --------------------------------------------------------------------------------------------------------------------
# Imaginary time:
# --------------------------------------------------------------------------------------------------------------------
for i in range(Nt):
    # Kinetic evolution:
    sm.first_kinetic_rot_evo(Psi, X, Y, Kx, Ky, Kz, omega_rot, spin_f, q, dt)

    # Non-linear evolution:
    Psi = sm.nonlin_evo(Psi[0], Psi[1], Psi[2], Psi[3], Psi[4], c0, c2, c4, V, p, dt, spin_f)

    # Kinetic evolution:
    sm.last_kinetic_rot_evo(Psi, X, Y, Kx, Ky, Kz, omega_rot, spin_f, q, dt)

    # Renormalise  atom number and fix phase:
    for ii in range(len(Psi)):
        Psi[ii] = cp.fft.fftn(cp.sqrt(N[ii]) * cp.fft.ifftn(Psi[ii]) / cp.sqrt(cp.sum(abs(cp.fft.ifftn(Psi[ii])) ** 2)))

    if i % 100 == 0:
        print('t = {}'.format(i * dt))

with h5py.File('test_data.hdf5', 'w') as file:
    file.create_dataset('wavefunction/psiP2', data=cp.asnumpy(cp.fft.ifftn(Psi[0])))
    file.create_dataset('wavefunction/psiP1', data=cp.asnumpy(cp.fft.ifftn(Psi[1])))
    file.create_dataset('wavefunction/psi0', data=cp.asnumpy(cp.fft.ifftn(Psi[2])))
    file.create_dataset('wavefunction/psiM1', data=cp.asnumpy(cp.fft.ifftn(Psi[3])))
    file.create_dataset('wavefunction/psiM2', data=cp.asnumpy(cp.fft.ifftn(Psi[4])))
