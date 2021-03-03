import odl
from enum import Enum 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class DataType(Enum):
    PHANTOM = 0
    CT = 1

class SolverType(Enum):
    STEEPEST_DECENT = 0
    DOUGLAS_RACHFORD_PD = 1


class CtSolver():
    def __init__(self, n=256, type=DataType.PHANTOM):
        self.n = n
        if type == DataType.PHANTOM:
            self.set_up_phantom()
        else:
            self.set_up_ct()
        
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
        
        self.ray_trafo = odl.tomo.RayTransform(
            self.input_space, self.geometry, impl="astra_cuda")
        self.output_space = self.ray_trafo.range

        self.g = self.ray_trafo(self.f_true)
        self.g_noisy = self.g
        

        
    def set_up_ct(self):
        #set the space
        self.input_space = odl.uniform_discr(
            [-112, -112, 0], [112, 112, 224], shape=(448, 448, 448), dtype='float32')
        #set the geometry
        #self.geometry = odl.contrib.tomo.elekta_icon_geometry()
        self.geometry = elekta_icon_geometry()
        #no grand truth image so we set this to 0 
        self.f_true = self.input_space.zero()
        self.ray_trafo = odl.tomo.RayTransform(
            self.input_space, self.geometry, impl="astra_cuda")
        self.output_space = self.ray_trafo.range
        self.g_noisy = np.load('/hl2027/noisy_data.npy')

    def add_noise(self, noise):
        self.noise = noise
        self.g_noisy = self.g + noise

    def save_error(self, f):
        self.error.append((self.f_true-f).norm())

    def solve_douglas_rachford_pd(self, lam=0.01, gamma=0.01, tau=1.0, niter=100 , verbose=True):
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
        if not verbose:
            odl.solvers.douglas_rachford_pd(
                self.x_drpd, f, g_funcs, lin_ops, tau=tau, sigma=sigma, niter=niter)
        else:
            odl.solvers.douglas_rachford_pd(
                self.x_drpd, f, g_funcs, lin_ops, tau=tau, sigma=sigma, niter=niter, callback=self.callback)

        return self.x_drpd, self.error

    def solve_stepest_decent(self, lam=0.01, gamma=0.01, line_search = 0.001, maxiter=300, verbose=True):
        self.error = []
        grad = odl.Gradient(self.input_space)
        huber_solver = odl.solvers.Huber(grad.range, gamma=gamma)  # small gamma

        Q = odl.solvers.L2NormSquared(self.output_space).translated(self.g_noisy) * self.ray_trafo + lam * huber_solver * grad

        self.x_sd = self.ray_trafo.domain.zero()
        if not verbose:
            odl.solvers.smooth.gradient.steepest_descent(
                Q, self.x_sd, line_search=line_search, maxiter=maxiter)
        else: 
            odl.solvers.smooth.gradient.steepest_descent(
                Q, self.x_sd, line_search=line_search, maxiter=maxiter, callback=self.callback)

        return self.x_sd, self.error

    def set_gamma(self, solver=SolverType.DOUGLAS_RACHFORD_PD):
        gammas = np.linspace(0.0001, 0.01, num=25)
        res_error = []
        for g in tqdm(gammas):
            if solver == SolverType.DOUGLAS_RACHFORD_PD:
                res_error.append((self.solve_douglas_rachford_pd(
                    gamma=g, verbose=False)[0] - self.f_true).norm())
            else:
                res_error.append((self.solve_stepest_decent(
                    gamma=g, verbose=False, maxiter=100)[0] - self.f_true).norm())
            
        plt.scatter(gammas, res_error)
        plt.xlabel("gamma")
        plt.ylabel("residual error")
        plt.title("Residual Error vs gamma of Huber Norm.")
        plt.grid(True)
        plt.show()

        return gammas, res_error

    
    
    """Tomography helpers for Elekta systems."""

def elekta_icon_geometry(sad=780.0, sdd=1000.0,
                         piercing_point=(390.0, 0.0),
                         angles=None, num_angles=None,
                         detector_shape=(780, 720)):
    """Tomographic geometry of the Elekta Icon CBCT system.
    See the [whitepaper]_ for specific descriptions of each parameter.
    All measurments are given in millimeters unless otherwise stated.
    Parameters
    ----------
    sad : float, optional
        Source to Axis distance.
    sdd : float, optional
        Source to Detector distance.
    piercing_point : sequence of foat, optional
        Position in the detector (in pixel coordinates) that a beam from the
        source, passing through the axis of rotation perpendicularly, hits.
    angles : array-like, optional
        List of angles given in radians that the projection images were taken
        at. Exclusive with num_angles.
        Default: np.linspace(1.2, 5.0, 332)
    num_angles : int, optional
        Number of angles. Exclusive with angles.
        Default: 332
    detector_shape : sequence of int, optional
        Shape of the detector (in pixels). Useful if a sub-sampled system
        should be studied.
    Returns
    -------
    elekta_icon_geometry : `ConeBeamGeometry`
    Examples
    --------
    Create default geometry:
    >>> from odl.contrib import tomo
    >>> geometry = tomo.elekta_icon_geometry()
    Use a smaller detector (improves efficiency):
    >>> small_geometry = tomo.elekta_icon_geometry(detector_shape=[100, 100])
    See Also
    --------
    elekta_icon_space : Default reconstruction space for the Elekta Icon CBCT.
    elekta_icon_fbp: Default reconstruction method for the Elekta Icon CBCT.
    References
    ----------
    .. [whitepaper] *Design and performance characteristics of a Cone Beam
       CT system for Leksell Gamma Knife Icon*
    """
    sad = float(sad)
    assert sad > 0
    sdd = float(sdd)
    assert sdd > sad
    piercing_point = np.array(piercing_point, dtype=float)
    assert piercing_point.shape == (2,)

    if angles is not None and num_angles is not None:
        raise ValueError('cannot provide both `angles` and `num_angles`')
    elif angles is not None:
        angles = odl.nonuniform_partition(angles)
        assert angles.ndim == 1
    elif num_angles is not None:
        angles = odl.uniform_partition(1.2, 5.0, num_angles)
    else:
        angles = odl.uniform_partition(1.2, 5.0, 332)

    detector_shape = np.array(detector_shape, dtype=int)

    # Constant system parameters
    pixel_size = 0.368
    det_extent_mm = np.array([287.04, 264.96])

    # Compute the detector partition
    piercing_point_mm = pixel_size * piercing_point
    det_min_pt = -piercing_point_mm
    det_max_pt = det_min_pt + det_extent_mm
    detector_partition = odl.uniform_partition(min_pt=det_min_pt,
                                               max_pt=det_max_pt,
                                               shape=detector_shape)

    # Create the geometry
    geometry = odl.tomo.ConeBeamGeometry(
        angles, detector_partition,
        src_radius=sad, det_radius=sdd - sad)

    return geometry
