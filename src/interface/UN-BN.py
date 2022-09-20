import numpy as np
import cupy as cp
import include.symplectic as sm
import h5py

# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny, Nz = 128, 128, 128  # Number of grid points
Mx, My, Mz = Nx // 2, Ny // 2, Nz // 2
dx, dy, dz = 20 / Nx, 20 / Ny, 20 / Nz  # Grid spacing
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
omega_trap = 1
omega_rot = 0.
V = 0.5 * omega_trap ** 2 * (X ** 2 + Y ** 2 + Z ** 2)
p = 0  # Linear Zeeman
sigma = 2.5
scale = 1
q = scale * (1 / (1 + np.exp(-sigma * Z)) - 0.5)

c0 = 1.32e4
c2 = 146
c4 = -129

# Time steps, number and wavefunction save variables
Nt = 50000
Nframe = 250  # Saves data every Nframe time steps
dt = -1j * 1e-2  # Time step
t = 0.

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
phi = cp.arctan2(Y - 0.01, X - 0.01)  # Phase is azimuthal angle around the core

Tf = sm.get_TF_density_3d(c0, c2, X, Y, Z, N=1)

eta = (1 / (1 + cp.exp(-sigma * Z)))

# Generate initial wavefunctions:
psiP2 = cp.sqrt(Tf) * cp.exp(1j * phi) * cp.sqrt((1 - eta ** 2) / 2)
psiP1 = cp.zeros((Nx, Ny, Nz))
psi0 = cp.sqrt(Tf) * cp.exp(1j * phi) * eta
psiM1 = cp.zeros((Nx, Ny, Nz))
psiM2 = cp.sqrt(Tf) * cp.exp(1j * phi) * cp.sqrt((1 - eta ** 2) / 2)

Psi = [psiP2, psiP1, psi0, psiM1, psiM2]  # Full 5x1 wavefunction

# Spin rotation on wavefunction:
alpha_angle = 0.01
beta_angle = 0.01
gamma_angle = 0

Psi = sm.rotation(Psi, alpha_angle, beta_angle, gamma_angle)
N = [dx * dy * cp.sum(cp.abs(wfn) ** 2) for wfn in Psi]  # Atom number of each component
theta_fix = [cp.angle(wfn) for wfn in Psi]
Psi = [cp.fft.fftn(wfn) for wfn in Psi]  # Transforming wfn to Fourier space

# Helper parameters for kinetic evolution
Ek = 0.5 * (Kx ** 2 + Ky ** 2 + Kz ** 2)
A = cp.exp(-1j * Ek * dt / 2)
B = cp.exp(-1j * Ek * dt / 2)
C = cp.exp(-1j * Ek * dt / 2)

# Store parameters in dictionary for saving
parameters = {
    "c0": c0,
    "c2": c2,
    "c4": c4,
    "q": q,
    "p": p,
    "omega_trap": omega_trap,
    "alpha, beta, gamma": [alpha_angle, beta_angle, gamma_angle]
}

# Create dataset and save initial state
filename = 'UN-BN_SQV-SQV'  # Name of file to save data to
data_path = '../../data/3D/{}.hdf5'.format(filename)
k = 0  # Array index

with h5py.File(data_path, 'w') as data:
    # Saving spatial data:
    data.create_dataset('grid/x', x.shape, data=cp.asnumpy(x))
    data.create_dataset('grid/y', y.shape, data=cp.asnumpy(y))
    data.create_dataset('grid/z', z.shape, data=cp.asnumpy(z))

    # Saving time variables:
    data.create_dataset('time/Nt', data=Nt)
    data.create_dataset('time/dt', data=dt)
    data.create_dataset('time/Nframe', data=Nframe)

    # Save parameters:
    data.create_dataset('parameters', data=str(parameters))

    # Creating empty wavefunction datasets to store data:
    data.create_dataset('wavefunction/psiP2', (Nx, Ny, Nz, 1), maxshape=(Nx, Ny, Nz, None), dtype='complex64')
    data.create_dataset('wavefunction/psiP1', (Nx, Ny, Nz, 1), maxshape=(Nx, Ny, Nz, None), dtype='complex64')
    data.create_dataset('wavefunction/psi0', (Nx, Ny, Nz, 1), maxshape=(Nx, Ny, Nz, None), dtype='complex64')
    data.create_dataset('wavefunction/psiM1', (Nx, Ny, Nz, 1), maxshape=(Nx, Ny, Nz, None), dtype='complex64')
    data.create_dataset('wavefunction/psiM2', (Nx, Ny, Nz, 1), maxshape=(Nx, Ny, Nz, None), dtype='complex64')

    # Store initial state
    data.create_dataset('initial_state/psiP2', data=cp.asnumpy(cp.fft.ifftn(Psi[0])))
    data.create_dataset('initial_state/psiP1', data=cp.asnumpy(cp.fft.ifftn(Psi[1])))
    data.create_dataset('initial_state/psi0', data=cp.asnumpy(cp.fft.ifftn(Psi[2])))
    data.create_dataset('initial_state/psiM1', data=cp.asnumpy(cp.fft.ifftn(Psi[3])))
    data.create_dataset('initial_state/psiM2', data=cp.asnumpy(cp.fft.ifftn(Psi[4])))

# --------------------------------------------------------------------------------------------------------------------
# Imaginary time:
# --------------------------------------------------------------------------------------------------------------------
for i in range(Nt):
    if i == 200:
        gamma = 1e-2
        dt = (1 - gamma * 1j) * 5e-3

    # Kinetic evolution:
    sm.first_kinetic_rot_evo_3d(Psi, X, Y, Kx, Ky, Kz, omega_rot, spin_f, dt)

    # Non-linear evolution:
    Psi = sm.nonlin_evo(Psi[0], Psi[1], Psi[2], Psi[3], Psi[4], c0, c2, c4, V, p, q, dt, spin_f)

    # Kinetic evolution:
    sm.last_kinetic_rot_evo_3d(Psi, X, Y, Kx, Ky, Kz, omega_rot, spin_f, dt)

    # Renormalise  atom number:
    for ii in range(len(Psi)):
        Psi[ii] = cp.fft.fftn(cp.sqrt(N[ii]) * cp.fft.ifftn(Psi[ii]) / cp.sqrt(cp.sum(abs(cp.fft.ifftn(Psi[ii])) ** 2)))

    if i % 100 == 0:
        print('t = {}'.format(i * dt))

    # Saves data
    if np.mod(i + 1, Nframe) == 0:
        # Updates file with new wavefunction values:
        with h5py.File(data_path, 'r+') as data:
            new_psiP2 = data['wavefunction/psiP2']
            new_psiP2.resize((Nx, Ny, Nz, k + 1))
            new_psiP2[:, :, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[0]))

            new_psiP1 = data['wavefunction/psiP1']
            new_psiP1.resize((Nx, Ny, Nz, k + 1))
            new_psiP1[:, :, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[1]))

            new_psi0 = data['wavefunction/psi0']
            new_psi0.resize((Nx, Ny, Nz, k + 1))
            new_psi0[:, :, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[2]))

            new_psiM1 = data['wavefunction/psiM1']
            new_psiM1.resize((Nx, Ny, Nz, k + 1))
            new_psiM1[:, :, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[3]))

            new_psiM2 = data['wavefunction/psiM2']
            new_psiM2.resize((Nx, Ny, Nz, k + 1))
            new_psiM2[:, :, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[4]))

        k += 1  # Increment array index
