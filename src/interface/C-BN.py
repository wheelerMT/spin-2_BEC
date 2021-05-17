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
omega_trap = 1
V = 0.5 * omega_trap ** 2 * (X ** 2 + Y ** 2 + Z ** 2)
p = 0.  # Linear Zeeman
q = -0.05  # Quadratic Zeeman
c0 = 5000
c2 = 1000
c4 = np.where(Z <= 0, 1000, -1000)

# Time steps, number and wavefunction save variables
Nt = 2500
Nframe = 50  # Saves data every Nframe time steps
dt = -1j * 1e-2  # Time step
t = 0.

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
phi = cp.arctan2(Y, X)  # Phase is azimuthal angle around the core

Tf = sm.get_TF_density(c0, c2, V)

eta = np.where(Z <= 0, 0, 1)  # Parameter used to interpolate between states

# Generate initial wavefunctions:
psiP2 = cp.sqrt(Tf) * cp.sqrt((1 + eta ** 2)) / 2
psiP1 = cp.zeros((Nx, Ny, Nz))
psi0 = cp.sqrt(Tf) * 1j * cp.sqrt((1 - eta ** 2) / 2)
psiM1 = cp.zeros((Nx, Ny, Nz))
psiM2 = cp.sqrt(Tf) * cp.sqrt((1 + eta ** 2)) / 2

Psi = [psiP2, psiP1, psi0, psiM1, psiM2]  # Full 5x1 wavefunction

# Spin rotation on wavefunction:
alpha_angle = 0
beta_angle = 0.1
gamma_angle = 0

Psi = sm.rotation(Psi, alpha_angle, beta_angle, gamma_angle)
N = [dx * dy * cp.sum(cp.abs(wfn) ** 2) for wfn in Psi]  # Atom number of each component
theta_fix = [cp.angle(wfn) for wfn in Psi]
Psi = [cp.fft.fftn(wfn) for wfn in Psi]  # Transforming wfn to Fourier space

# Store parameters in dictionary for saving
parameters = {
    "c0": c0,
    "c2": c2,
    "c4": c4,
    "q": q,
    "p": p,
    "omega_trap": omega_trap,
    "omega_rot": omega_rot,
    "alpha, beta, gamma": [alpha_angle, beta_angle, gamma_angle]
}

# Create dataset and save initial state
filename = 'C-BN_interface'  # Name of file to save data to
data_path = '../../data/{}.hdf5'.format(filename)
k = 0  # Array index

with h5py.File(data_path, 'w') as data:
    # Saving spatial data:
    data.create_dataset('grid/x', x.shape, data=cp.asnumpy(x))
    data.create_dataset('grid/y', y.shape, data=cp.asnumpy(y))

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
    # Kinetic evolution:
    sm.first_kinetic_rot_evo_3d(Psi, X, Y, Kx, Ky, Kz, omega_rot, spin_f, q, dt)

    # Non-linear evolution:
    Psi = sm.nonlin_evo(Psi[0], Psi[1], Psi[2], Psi[3], Psi[4], c0, c2, c4, V, p, dt, spin_f)

    # Kinetic evolution:
    sm.last_kinetic_rot_evo_3d(Psi, X, Y, Kx, Ky, Kz, omega_rot, spin_f, q, dt)

    # Renormalise  atom number and fix phase:
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
