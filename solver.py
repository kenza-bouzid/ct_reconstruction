import odl
from enum import Enum 
import numpy as np


class DataType(Enum):
    PHANTOM = 0
    CT = 1


class CtSolver():
    def __init__(self, n=256, type=DataType.PHANTOM):
        self.n = n
        if type == DataType.PHANTOM:
            self.set_up_phantom()
        else:
            self.set_up_ct()
        
        self.ray_trafo = odl.tomo.RayTransform(self.input_space, self.geometry, impl="astra_cuda")
        self.g = self.ray_trafo(self.f_true)
        self.g_noisy = self.g
        self.output_space = self.ray_trafo.range

        self.error = []

        self.callback = (odl.solvers.CallbackPrintIteration(step=10) &
                    odl.solvers.CallbackShow(step=10) &
                    self.save_error)

    def set_up_phantom(self):
        #set the space 
        xlim = 10
        self.input_space = odl.uniform_discr(
            min_pt=[-xlim]*2, max_pt=[xlim]*2, shape=(self.n, self.n))
        #set the geometry
        apart = odl.uniform_partition(0, 2*np.pi, self.n)  
        dpart = odl.uniform_partition(-3*xlim, 3*xlim, self.n)
        self.geometry = odl.tomo.FanBeamGeometry(apart=apart, dpart=dpart, src_radius=2*xlim, det_radius=2*xlim)
        #set the data 
        self.f_true = odl.phantom.shepp_logan(self.input_space, modified=True)

        
    def set_up_ct(self):
        #set the space
        self.input_space = odl.uniform_discr(
            [-112, -112, 0], [112, 112, 224], shape=(448, 448, 448), dtype='float32')
        #set the geometry
        self.geometry = odl.contrib.tomo.elekta_icon_geometry()
        #set the data
        self.f_true = self.input_space.element(
            np.load('/hl2027/noisy data.npy'))

    def add_noise(self, noise):
        self.noise = noise
        self.g_noisy = self.g + noise

    def save_error(self, f):
        self.error.append((self.f_true-f).norm())

    def solve_douglas_rachford_pd(self, lam=0.01, gamma=0.01, tau=1.0, niter=100):
        self.error = []
        # Assemble all operators into a list
        grad = odl.Gradient(self.input_space)

        lin_ops = [self.ray_trafo, grad]

        # Create functionals for the l2 distance and huber norm
        g_funcs = [odl.solvers.L2NormSquared(self.output_space).translated(self.g_noisy),
                   lam * odl.solvers.Huber(grad.range, gamma=gamma)]

        #Functional of the bound contraint 0 <= x <= 1
        f = odl.solvers.IndicatorBox(self.input_space, 0, 1)

        # Find scaling constants so that the solver converges.
        opnorm_ray_trafo = odl.power_method_opnorm(self.ray_trafo, xstart=self.g_noisy)
        opnorm_grad = odl.power_method_opnorm(grad, xstart=self.g_noisy)
        sigma = [1 / opnorm_ray_trafo ** 2, 1 / opnorm_grad ** 2]
        tau = 1.0

        # Solve using the Douglas-Rachford Primal-Dual method
        self.x_drpd = self.input_space.zero()

        odl.solvers.douglas_rachford_pd(
            self.x_drpd, f, g_funcs, lin_ops, tau=tau, sigma=sigma, niter=niter, callback=self.callback)

        return self.x_drpd, self.error

    def solve_stepest_decent(self, lam=0.01, gamma=0.01, line_search = 0.001, maxiter=300):
        self.error = []
        grad = odl.Gradient(self.input_space)
        huber_solver = odl.solvers.Huber(grad.range, gamma=gamma)  # small gamma

        Q = odl.solvers.L2NormSquared(self.output_space).translated(self.g_noisy) * self.ray_trafo + lam * huber_solver * grad

        self.x_sd = self.ray_trafo.domain.zero()
        odl.solvers.smooth.gradient.steepest_descent(
            Q, self.x_sd, line_search=line_search, maxiter=maxiter, callback=self.callback)

        return self.x_sd, self.error

