from gpe3d import gpe3d_python
import numpy as np
from common import Prob, saveobject, loadobject, to_nK
from rbconstants import *
import matplotlib.pyplot as plt


class harmonic_isotropic_3d(object):

    """Finds the ground state of a BEC in a harmonic and isotropic trap."""

    f = 30  # Hz trap frequency
    n_atoms = 300e3  # number of atoms
    dt = 10e-6  # (s) time step
    dx = 400e-9  # (m) spatial step
    t_prop_imag = 30e-3  # (s)
    size = 7
    nt_store = 10

    def __init__(self):
        self.nt_imag = int(self.t_prop_imag / self.dt)
        self.omega = 2.0 * pi * self.f

        self.energy_scale = hbar * self.omega
        self.time_scale = 1.0 / self.omega
        self.length_scale = (hbar * self.time_scale / m_rb) ** 0.5

        self.dy = self.dx
        self.dz = self.dx

        self.nx = 2 ** self.size
        self.ny = self.nx
        self.nz = self.nx

        self.x = (np.arange(0, self.nx)-(self.nx - 1) / 2.0) * self.dx
        self.y = (np.arange(0, self.ny)-(self.ny - 1) / 2.0) * self.dy
        self.z = (np.arange(0, self.nz)-(self.nz - 1) / 2.0) * self.dz

        self.x_nd = self.x / self.length_scale
        self.y_nd = self.y / self.length_scale
        self.z_nd = self.z / self.length_scale

        self.dt_nd = self.dt / self.time_scale

        self.kappa = 4.0 * pi * a_bg * self.n_atoms / self.length_scale

        xg, yg, zg = np.meshgrid(self.x_nd, self.y_nd, self.z_nd)
        self.calculate_analytical()
        self.r_squared_nd = xg ** 2 + yg ** 2 + zg ** 2
        rad_nd = 3e-6 / self.length_scale
        self.U_nd = 0.5 * self.r_squared_nd
        # self.psi0 = np.exp(-self.r_squared_nd / rad_nd ** 2).astype(complex)
        parabola = (self.mu_analytical_nd - 0.5 * self.r_squared_nd) / self.kappa
        self.psi0 = ((parabola > 0.0) * parabola) ** 0.5

    def calculate_analytical(self):
        self.mu_analytical_nd = 0.5 * (15.0 * self.n_atoms * a_bg /
                                       self.length_scale) ** (2.0/5.0)
        self.r_tf_analytical_nd = (2.0 * self.mu_analytical_nd) ** 0.5
        self.mu_analytical = self.mu_analytical_nd * self.energy_scale
        self.r_tf_analytical = self.r_tf_analytical_nd * self.length_scale

    def info_string(self):
        strlist = ['Time scale : {0:.3f} ms'.format(self.time_scale * 1e3),
                   'Length scale : {0:.3f} um'.format(self.length_scale * 1e6),
                   'Energy scale: {0:.3f} nK'.format(self.energy_scale * 1e9/ kb),
                   'Imag time steps : {0}'.format(self.nt_imag),
                   'Total length : {0:3f} um'.format(self.x[0] * -2.0 * 1e6),
                   'TF Radius {0:3f} um'.format(self.r_tf_analytical * 1e6),
                   'Chemical Potential {0:3f} nK'.format(to_nK(self.mu_analytical))]
        return '\n'.join(strlist)

    def run_imag(self):
        output = gpe3d_python(self.kappa, self.nt_imag, self.dt_nd,
                              self.x_nd, self.y_nd, self.z_nd,
                              self.U_nd, self.psi0,
                              self.nt_store, imag_time=1)

        (self.kx_nd, self.ky_nd, self.kz_nd, self.t_stored,
         self.psi_ground) = output

    def calculate_ground_state_properties(self):
        self.dkx_nd = self.kx_nd[1] - self.kx_nd[0]
        self.dky_nd = self.ky_nd[1] - self.ky_nd[0]
        self.dkz_nd = self.kz_nd[1] - self.kz_nd[0]

        self.dx_nd = self.x_nd[1] - self.x_nd[0]
        self.dy_nd = self.y_nd[1] - self.y_nd[0]
        self.dz_nd = self.z_nd[1] - self.z_nd[0]

        self.dV_nd = self.dx_nd * self.dy_nd * self.dz_nd
        self.dVk_nd = self.dkx_nd * self.dky_nd * self.dkz_nd

        (self.kx_grid_nd,
         self.ky_grid_nd,
         self.kz_grid_nd) = np.meshgrid(self.kx_nd, self.ky_nd, self.kz_nd)
        self.k_squared_nd = (self.kx_grid_nd ** 2 + self.ky_grid_nd ** 2 +
                             self.kz_grid_nd ** 2)

        # find K space wavefunction
        self.psi_ground_k = np.fft.fftn(self.psi_ground)
        sum_k = np.sum(Prob(self.psi_ground_k)) * self.dVk_nd
        self.psi_ground_k /= sum_k ** 0.5

        self.p2x = Prob(self.psi_ground)
        self.p2k = Prob(self.psi_ground_k)

        self.kinetic_ex = (np.sum(self.k_squared_nd * self.p2k) *
                           (self.dVk_nd / 2.0 * self.energy_scale))
        self.potential_ex = (np.sum(self.U_nd * self.p2x) *
                             (self.dV_nd * self.energy_scale))
        self.mfe_ex = (np.sum(self.p2x ** 2) *
                       self.dV_nd * self.kappa * self.energy_scale)
        self.mu_ex = self.kinetic_ex + self.potential_ex + self.mfe_ex

if __name__ == '__main__':
    sim = harmonic_isotropic_3d()
    print(sim.info_string())
    # sim.run_imag()
