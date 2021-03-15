import odl
from enum import Enum 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class DataType(Enum):
    """Class enumeration to decide whether to work with the phantom or the CT data
    
    """
    PHANTOM = 0
    CT = 1

class SolverType(Enum):
    """Class enumeration to decide whether to use Douglas Rachford or Steepest Descent metod
    
    """
    STEEPEST_DECENT = 0
    DOUGLAS_RACHFORD_PD = 1


class CtSolver():
    """Class to solve the reconstruction problem, phantom experiment to the real CT-data reconstruction

    Methods:
        set_up_phantom: Setting up environment for the phantom (input space, geometry, ray transform, output range)
        set_up_ct: Setting up environment for the CT acquisition (input space, geometry, ray transform, output range)
        add_noise: Add the noise to the sinogram obtained by applying the ray transform to the ground truth phantom
        save_error: Add the computed error, norm of the difference between ground truth and estimated image, each iteration step. 
                    Called in the callback
        solve_douglas_rachford_pd: Douglas Rachford Primer Dual Solver method (defining also the gradient operator and the functionals)
        solve_stepest_decent: Steepest descent method (defining also the gradient operator and the functionals)
        set_gamma: Choose the gamma for the Huber function so that the residual error is minimized.
                   The plot of gammas against residual error is displayed
    """
    def __init__(self, n=256, type=DataType.PHANTOM):
        """Define if working with phantom or CT-data. Saving the error during iterations with callback

        Args:
            n (int, optional): number of sample for image space discretization. Defaults to 256.
            type (int, optional): input data from DataType class. Defaults to DataType.PHANTOM.

        Attributes:
            error(list): will collect the errors (norm of difference: ground truth - estimated image) computed during the iterations.
            callback(): will print the iteration count, show the image and save the error (when verbose=True in the solver)
        """
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
        """Setting up environment for the phantom: input space, geometry, ray transform, output range
        
        Attributes:
            input_space (ODL element): uniformly discretized L^p function space for the phantom
            geometry (ODL element): Tomographic Fan Beam Geometry
            f_true (ODL element): ground truth set as the Shepp Logan phantom
            ray_trafo(ODL element): ray transform defined with the geometry and the input_space
            output_space(ODL element): range of the ray transform, sinogram domain
            g(ODL element): sinogram
            g_noisy(ODL element): noisy sinogram initialized without noise

        Variables:
            apart (ODL element): Partition of the angle interval
            dpart (ODL element): Partition of the detector parameter interval

        """   
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
        """Setting up the environment for the CT acquisition: input space, geometry, ray transform, output range
        
        Attributes:
            input_space (ODL element): uniformly discretized L^p function space for the CT image
            geometry (ODL element): Tomographic geometry of the Elekta Icon CBCT system
            f_true (ODL element): ground truth set to zero
            ray_trafo(ODL element): ray transform defined with the geometry and the input_space
            output_space(ODL element): range of the ray transform, data domain
            g_noisy(numpy.ndarray): loaded data

        """
        
        #set the space
        self.input_space = odl.uniform_discr(
            [-112, -112, 0], [112, 112, 224], shape=(448, 448, 448), dtype='float32')
        
        #set the geometry
        self.geometry = elekta_icon_geometry()

        #no grand truth image so we set this to 0 
        self.f_true = self.input_space.zero()
        
        self.ray_trafo = odl.tomo.RayTransform(
            self.input_space, self.geometry, impl="astra_cuda")
        
        self.output_space = self.ray_trafo.range
        
        data = np.load('/hl2027/noisy_data.npy')
        self.g_noisy = self.output_space.element(data)
        

    def add_noise(self, noise):
        """Add the noise to the sinogram obtained by applying the ray transform to the ground truth phantom

        Args:
            noise (float): noise amplification coefficient multiplying by the white noise of ODL
        Attributes:
            noise (float): noise as an attribute
            g_noisy(ODL element): noisy sinogram
        """
        self.noise = noise
        self.g_noisy = self.g + noise * odl.phantom.white_noise(self.output_space)

    def save_error(self, f):
        """Add the computed error, norm of the difference between ground truth and estimated image, each iteration step. 
            Called in the callback

        Args:
            f (ODL element): estimated image
        """
        self.error.append((self.f_true-f).norm())

    def solve_douglas_rachford_pd(self, lam=0.01, gamma=0.01, tau=1.0, niter=100 , verbose=True):
        """Douglas Rachford Primer Dual Solver method (defining also the gradient operator and the functionals)

        Args:
            lam (float, optional): lambda parameter for regularization. Defaults to 0.01.
            gamma (float, optional): gamma parameter for the Huber function. Defaults to 0.01.
            tau (float, optional): step size parameter. Defaults to 1.0.
            niter (int, optional): number of iterations. Defaults to 100.
            verbose (bool, optional): if True the callback will be enabled. Defaults to True.

        Returns:
            x_drpd(ODL element): reconstructed image (solution of the problem)
            error(list):list with the computed errors at each iteration

        """
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
        opnorm_ray_trafo = self.ray_trafo.norm(estimate=True)
        opnorm_grad = grad.norm(estimate=True)
        sigma = [1 / opnorm_ray_trafo ** 2, 1 / opnorm_grad ** 2]
        
        
        # Solve using the Douglas-Rachford Primal-Dual method
        self.x_drpd = self.input_space.zero()
        
        if not verbose:
            odl.solvers.douglas_rachford_pd(
                self.x_drpd, f, g_funcs, lin_ops, tau=tau, sigma=sigma, niter=niter)
        else:
            odl.solvers.douglas_rachford_pd(
                self.x_drpd, f, g_funcs, lin_ops, tau=tau, sigma=sigma, niter=niter, callback=self.callback)

        return self.x_drpd, self.error
    

    def solve_stepest_decent(self, lam=0.01, gamma=0.001, maxiter=300, control=0.1, verbose=True):
        """Steepest descent method (defining also the gradient operator and the functionals)

        Args:
            lam (float, optional): lambda parameter for regularization. Defaults to 0.01.
            gamma (float, optional): gamma parameter for the Huber function. Defaults to 0.001.
            maxiter (int, optional): number of iterations. Defaults to 300.
            control (float, optional): To estimate line search, which is step length. Defaults to 0.1.
            verbose (bool, optional): if True the callback will be enabled. Defaults to True.

        Returns:
            x_sd(ODL element): reconstructed image (solution of the problem)
            error(list): list with the computed errors at each iteration

        """
        self.error = []
        
        # Create functionals
        grad = odl.Gradient(self.input_space)
        huber_solver = odl.solvers.Huber(grad.range, gamma=gamma)  # small gamma
        
        #Create goal functional
        Q1 = odl.solvers.L2NormSquared(self.output_space).translated(self.g_noisy) * self.ray_trafo
        Q2 = huber_solver * grad

        Q = Q1 + lam * Q2 
        
        #estimate learning_rate
        norm1 = self.ray_trafo.norm(estimate=True)**2
        norm2 = 1/gamma * grad.norm(estimate=True)**2
        
        line_search = 2/((norm1 + lam * norm2)) * control
        print(f"learning_rate:{line_search}")
        self.x_sd = self.ray_trafo.domain.zero()
        if not verbose:
            odl.solvers.smooth.gradient.steepest_descent(
                Q, self.x_sd, line_search=line_search, maxiter=maxiter)
        else: 
            odl.solvers.smooth.gradient.steepest_descent(
                Q, self.x_sd, line_search=line_search, maxiter=maxiter, callback=self.callback)

        return self.x_sd, self.error

    def set_gamma(self, solver=SolverType.DOUGLAS_RACHFORD_PD, _min=0.0001, _max=0.01):
        """Choose the gamma for the Huber function so that the residual error is minimized. 
           The plot of gammas against residual error is displayed

        Args:
            _min (float, optional): set the minimum lambda. Defaults to 0.0001
            _max (float, optional): set the maximum lambda. Defaults to 0.01
            solver (int, optional): set the solver to estimate gamma. Defaults to SolverType.DOUGLAS_RACHFORD_PD.

        Returns:
            gammas(numpy.ndarray): array containing gamma evenly spaced
            res_error(list): list containing the residual errors
        """
        #initialize gammas
        gammas = np.linspace(_min, _max, num=25)
        res_error = []
        for g in tqdm(gammas):
            if solver == SolverType.DOUGLAS_RACHFORD_PD:
                res_error.append((self.solve_douglas_rachford_pd(
                    gamma=g, verbose=False)[0] - self.f_true).norm())
            else:
                res_error.append((self.solve_stepest_decent(
                    gamma=g, verbose=False, maxiter=100)[0] - self.f_true).norm())
        #plot    
        plt.scatter(gammas, res_error)
        plt.xlabel("gamma")
        plt.ylabel("residual error")
        plt.title("Residual Error vs gamma of Huber Norm.")
        plt.grid(True)
        plt.show()

        return gammas, res_error

    
#Directly taken from the odl public repository:    
    
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
