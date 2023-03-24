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
dkx, dky, dkz = (
    np.pi / (Mx * dx),
    np.pi / (My * dy),
    np.pi / (Mz * dz),
)  # K-space spacing
len_x, len_y, len_z = Nx * dx, Ny * dy, Nz * dz  # Box length

# Generate 1-D spatial grids:
x = cp.arange(-Mx, Mx) * dx
y = cp.arange(-My, My) * dy
z = cp.arange(-Mz, Mz) * dz
X, Y, Z = cp.meshgrid(x, y, z, indexing="ij")

# Generate 1-D k-space grids:
kx = cp.arange(-Mx, Mx) * dkx
ky = cp.arange(-My, My) * dky
kz = cp.arange(-Mz, Mz) * dkz
Kx, Ky, Kz = cp.meshgrid(kx, ky, kz, indexing="ij")
Kx, Ky, Kz = cp.fft.fftshift(Kx), cp.fft.fftshift(Ky), cp.fft.fftshift(Kz)

# Controlled variables:
spin_f = 2  # Spin-2
omega_trap = 1
omega_rot = 0.0
V = 0.5 * omega_trap**2 * (X**2 + Y**2 + Z**2)
p = 0  # Linear Zeeman
q = -1

c0 = 1e4
c2 = -250
c4 = 1000

# Time steps, number and wavefunction save variables
Nt = 5000
Nframe = 50  # Saves data every Nframe time steps
dt = -1j * 1e-2  # Time step
t = 0.0

# --------------------------------------------------------------------------------------------------------------------
# Generating initial state:
# --------------------------------------------------------------------------------------------------------------------
phi = cp.arctan2(Y, X)  # Phase is azimuthal angle around the core

Tf = sm.get_TF_density_3d(c0, c2, X, Y, Z, N=1)

# Spin rotation on wavefunction:
r = cp.sqrt(X**2 + Y**2)
alpha_angle = 0
beta_angle = cp.pi / 2 * (1 + cp.tanh(r - 1))
gamma_angle = 0

C = cp.cos(beta_angle / 2)
S = cp.sin(beta_angle / 2)

# Generate initial wavefunctions:
psiP2 = cp.sqrt(Tf) * (C**4 + S**4)
psiP1 = cp.sqrt(Tf) * cp.exp(1j * phi) * 2 * C * S * (C**2 - S**2)
psi0 = cp.sqrt(Tf) * 2 * cp.sqrt(6) * cp.exp(2j * phi) * C**2 * S**2
psiM1 = cp.sqrt(Tf) * 2 * cp.exp(3j * phi) * C * S * (S**2 - C**2)
psiM2 = cp.sqrt(Tf) * cp.exp(4j * phi) * (S**4 + S**4)

Psi = [psiP2, psiP1, psi0, psiM1, psiM2]  # Full 5x1 wavefunction

# Psi = sm.rotation(Psi, alpha_angle, beta_angle, gamma_angle)
N = [
    dx * dy * cp.sum(cp.abs(wfn) ** 2) for wfn in Psi
]  # Atom number of each component
theta_fix = [cp.angle(wfn) for wfn in Psi]
Psi = [cp.fft.fftn(wfn) for wfn in Psi]  # Transforming wfn to Fourier space

# Helper parameters for kinetic evolution
Ek = 0.5 * (Kx**2 + Ky**2 + Kz**2)
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
}

# Create dataset and save initial state
filename = "BN_DQV"  # Name of file to save data to
data_path = "../../data/3D/{}.hdf5".format(filename)
k = 0  # Array index

with h5py.File(data_path, "w") as data:
    # Saving spatial data:
    data.create_dataset("grid/x", x.shape, data=cp.asnumpy(x))
    data.create_dataset("grid/y", y.shape, data=cp.asnumpy(y))
    data.create_dataset("grid/z", z.shape, data=cp.asnumpy(z))

    # Saving time variables:
    data.create_dataset("time/Nt", data=Nt)
    data.create_dataset("time/dt", data=dt)
    data.create_dataset("time/Nframe", data=Nframe)

    # Save parameters:
    for key, val in parameters.items():
        data.create_dataset(f"parameters/{key}", data=val)

    # Creating empty wavefunction datasets to store data:
    data.create_dataset(
        "wavefunction/psiP2",
        (Nx, Ny, Nz, 1),
        maxshape=(Nx, Ny, Nz, None),
        dtype="complex64",
    )
    data.create_dataset(
        "wavefunction/psiP1",
        (Nx, Ny, Nz, 1),
        maxshape=(Nx, Ny, Nz, None),
        dtype="complex64",
    )
    data.create_dataset(
        "wavefunction/psi0",
        (Nx, Ny, Nz, 1),
        maxshape=(Nx, Ny, Nz, None),
        dtype="complex64",
    )
    data.create_dataset(
        "wavefunction/psiM1",
        (Nx, Ny, Nz, 1),
        maxshape=(Nx, Ny, Nz, None),
        dtype="complex64",
    )
    data.create_dataset(
        "wavefunction/psiM2",
        (Nx, Ny, Nz, 1),
        maxshape=(Nx, Ny, Nz, None),
        dtype="complex64",
    )

    # Store initial state
    data.create_dataset(
        "initial_state/psiP2", data=cp.asnumpy(cp.fft.ifftn(Psi[0]))
    )
    data.create_dataset(
        "initial_state/psiP1", data=cp.asnumpy(cp.fft.ifftn(Psi[1]))
    )
    data.create_dataset(
        "initial_state/psi0", data=cp.asnumpy(cp.fft.ifftn(Psi[2]))
    )
    data.create_dataset(
        "initial_state/psiM1", data=cp.asnumpy(cp.fft.ifftn(Psi[3]))
    )
    data.create_dataset(
        "initial_state/psiM2", data=cp.asnumpy(cp.fft.ifftn(Psi[4]))
    )
