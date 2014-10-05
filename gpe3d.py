"""A module which provides numerical routines to solve the three dimensional
Gross-Pitaevskii equation."""

import numpy as np
import numpy.fft as fft


def gpe3d_python(kappa, Nt, dt, X, Y, Z, U,  psi0, Ntstore=10, imag_time=0):
    Ntskip = Nt / (Ntstore - 1)
    Nx, Ny, Nz = np.size(X), np.size(Y), np.size(Z)
    dx, dy, dz = (X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0])
    dV = dx * dy * dz
    Kx = fft.fftfreq(Nx, dx) * 2.0 * np.pi
    Ky = fft.fftfreq(Ny, dy) * 2.0 * np.pi
    Kz = fft.fftfreq(Nz, dz) * 2.0 * np.pi

    T = np.zeros(Ntstore)

    if imag_time == 0:
        prefactor = 1j
        psi_out = np.zeros((Ntstore, Nx, Ny, Nz), complex)
        psi_out[0, :] = psi0
    else:
        prefactor = 1
        psi_out = np.zeros((Nx, Ny, Nz), complex)

    U1 = -prefactor * U * dt / 2.0
    C1 = -prefactor * kappa * dt / 2.0
    Kxg, Kyg, Kzg = np.meshgrid(Kx, Ky, Kz)
    K_squared = Kxg ** 2 + Kyg ** 2 + Kzg ** 2
    Kin = np.exp(-prefactor * K_squared * dt / 2.0)
    psi = psi0

    i = 0
    for t1 in range(Ntstore-1):
        for t2 in range(Ntskip):
            print('step ' + str(i) + 'of ' + str(Nt))
            i += 1
            # Split the entire time stepping into three steps.
            # The first is stepping by time k/2 but only applying the potential
            # and the mean field parts of the unitary
            psi_squared = psi * np.conj(psi)
            psi = np.exp(U1 + C1*psi_squared) * psi
            print('first step')
            psi_int = np.sum(np.conj(psi) * psi) * dV
            print(psi_int)
            # The second part is applying the Kinetic part of the unitary. This
            # is done by taking the fourier transform of psi, so applying this
            # unitary in k space is simply multiplying it by another array
            psi = fft.ifftn(Kin*fft.fftn(psi))
            print('second step')
            psi_int = np.sum(np.conj(psi) * psi) * dV
            print(psi_int)
            # The third part is again stepping by k/2 and applying the
            # potential and interaction part of the unitary
            psi_squared = psi * np.conj(psi)
            psi = np.exp(U1 + C1*psi_squared) * psi
            if imag_time:
                # If we are propagating in imaginary time, then the solution
                # dies down, we need to explicitly normalize it
                print('third step')
                psi_int = np.sum(np.conj(psi) * psi) * dV
                print(psi_int)
                psi /= psi_int**0.5
                psi_int = np.sum(np.conj(psi) * psi) * dV
                print(psi_int)

        # Store the wavefuction in psi_out
        T[t1+1] = (t1+1) * dt * Ntskip
        if imag_time == 0:
            psi_out[t1+1, :] = psi
    if imag_time == 1:
        psi_out = psi
    return (Kx, Ky, Kz, T, psi_out)
