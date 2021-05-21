import numpy as np
import cupy as cp
import include.symplectic as sm
import h5py

# --------------------------------------------------------------------------------------------------------------------
# Spatial and Potential parameters:
# --------------------------------------------------------------------------------------------------------------------
Nx, Ny = 256, 256  # Number of grid points
Mx, My = Nx // 2, Ny // 2
dx, dy = 0.5, 0.5  # Grid spacing
dkx, dky = np.pi / (Mx * dx), np.pi / (My * dy)  # K-space spacing
len_x, len_y = Nx * dx, Ny * dy  # Box length

# Generate 1-D spatial grids:
x = cp.arange(-Mx, Mx) * dx
y = cp.arange(-My, My) * dy
X, Y = cp.meshgrid(x, y, indexing='ij')

# Generate 1-D k-space grids:
kx = cp.arange(-Mx, Mx) * dkx
ky = cp.arange(-My, My) * dky
Kx, Ky = cp.meshgrid(kx, ky, indexing='ij')
Kx, Ky = cp.fft.fftshift(Kx), cp.fft.fftshift(Ky)

# Controlled variables:
spin_f = 2  # Spin-2
V = 0.
p = 0  # Linear Zeeman
q = -0.05  # Quadratic Zeeman
c0 = 1
c2 = 0.5
c4 = -0.5

# Time steps, number and wavefunction save variables
Nt = 2500
Nframe = 50  # Saves data every Nframe time steps
dt = -1j * 1e-2  # Time step
t = 0.

# Kinetic subsystem coefficients:
Ek = Kx ** 2 + Ky ** 2
A = cp.exp(-1j * (Ek + 4 * q) * dt)
B = cp.exp(-1j * (Ek + q) * dt)
C = cp.exp(-1j * Ek * dt)

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
# Generate initial wavefunctions:
psiP2 = 1 / cp.sqrt(2) * cp.ones((Nx, Ny))
psiP1 = cp.zeros((Nx, Ny))
psi0 = cp.zeros((Nx, Ny))
psiM1 = cp.zeros((Nx, Ny))
psiM2 = 1 / cp.sqrt(2) * cp.ones((Nx, Ny))

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
    "alpha, beta, gamma": [alpha_angle, beta_angle, gamma_angle]
}

# Create dataset and save initial state
filename = '2d_untrapped'  # Name of file to save data to
data_path = '../../data/2D/{}.hdf5'.format(filename)
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
    data.create_dataset('wavefunction/psiP2', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')
    data.create_dataset('wavefunction/psiP1', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')
    data.create_dataset('wavefunction/psi0', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')
    data.create_dataset('wavefunction/psiM1', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')
    data.create_dataset('wavefunction/psiM2', (Nx, Ny, 1), maxshape=(Nx, Ny, None), dtype='complex64')

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
    Psi = sm.first_kinetic_evo(Psi, A, B, C)

    # Non-linear evolution:
    Psi = sm.nonlin_evo(Psi[0], Psi[1], Psi[2], Psi[3], Psi[4], c0, c2, c4, V, p, dt, spin_f)

    # Kinetic evolution:
    Psi = sm.last_kinetic_evo(Psi, A, B, C)

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
            new_psiP2.resize((Nx, Ny, k + 1))
            new_psiP2[:, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[0]))

            new_psiP1 = data['wavefunction/psiP1']
            new_psiP1.resize((Nx, Ny, k + 1))
            new_psiP1[:, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[1]))

            new_psi0 = data['wavefunction/psi0']
            new_psi0.resize((Nx, Ny, k + 1))
            new_psi0[:, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[2]))

            new_psiM1 = data['wavefunction/psiM1']
            new_psiM1.resize((Nx, Ny, k + 1))
            new_psiM1[:, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[3]))

            new_psiM2 = data['wavefunction/psiM2']
            new_psiM2.resize((Nx, Ny, k + 1))
            new_psiM2[:, :, k] = cp.asnumpy(cp.fft.ifftn(Psi[4]))

        k += 1  # Increment array index
