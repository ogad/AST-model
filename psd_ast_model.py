# Particle size distribution model
# Author: Oliver Driver
# Date: 01/06/2023

from random import choices
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from numpy.random import randint
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit


from ast_model import ASTModel


class CrystalModel(Enum):
    """Enum for crystal types."""
    SPHERE = 1
    RECT_AR5 = 2
    ROS_6 = 3

    def get_generator(self):
        if self == CrystalModel.SPHERE:
            return ASTModel.from_diameter
        elif self == CrystalModel.RECT_AR5:
            return lambda diameter, **kwargs: ASTModel.from_diameter_rectangular(diameter, 5, **kwargs)
        elif self == CrystalModel.ROS_6:
            return lambda diameter, **kwargs: ASTModel.from_diameter_rosette(diameter, 3, **kwargs)
        else:
            raise ValueError("Crystal model not recognised.")
    

def rejection_sampler(p, xbounds, pmax):
    """Returns a value sampled from a bounded probability distribution.

    Args:
        p (callable): The probability distribution function.
        xbounds (list): The bounds of the distribution.
        pmax (float): The maximum value of the probability distribution."""
    while True:
        # Generate a random x and y value
        x = np.random.rand(1) * (xbounds[1] - xbounds[0]) + xbounds[0]
        y = np.random.rand(1) * pmax

        # If the y value is below the probability distribution, return the x value
        if y <= p(x):
            return x

class PSD(ABC):
    """Base class for particle size distribution objects."""
    def __init__(self, bins: list[float] = None, model: CrystalModel = CrystalModel.SPHERE):
        if bins is None and getattr(self,"xlim", None) is not None:
            bins = np.logspace(np.log10(max(self.xlim[0], 10e-6)), np.log10(self.xlim[1]), 100)
        elif bins is None:
            bins = np.logspace(-7, -3, 100)
        self.bins = bins

        self.model = model

    @abstractmethod
    def dn_dd(self, d: ArrayLike) -> np.ndarray:
        """Calculate the particle size distribution value given diameters.
        """
        pass

    @property
    def binned_distribution(self):
        """Calculate the binned particle size distribution.

        Returns:
            np.ndarray: The number of particles in each bin.
        """
        return self.dn_dd(self.bins[1:]) * (np.diff(self.bins))
    
    def generate_diameters(self, n_particles) -> tuple[list[float], list[ASTModel]]:
        """Generate a particle diameter from the PSD."""
        diameters = choices(
                self.bins[1:], weights=self.binned_distribution, k=n_particles
            )
        return diameters, [self.model] * n_particles
    
    @property
    def total_number_density(self) -> float:
        """Calculate the total number density of particles."""
        # Uses the analytical expression for the integral of the gamma distribution
        # return self.intercept * self.slope ** (-self.shape - 1) * np.math.gamma(self.shape + 1) 
        return self.binned_distribution.sum()
    
    def plot(self, ax, **kwargs):
        """Plot the PSD value against diameter."""
        ax.plot(self.bins[1:], self.dn_dd(self.bins[1:]), **kwargs)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel('Diameter (m)')
        ax.set_ylabel('PSD (m$^{-3}$)')

class CompositePSD(PSD):
    """A composite particle size distribution object.

    Contains the distribution parameters, and binning information.

    Args:
        psds (list[PSD]): The list of PSDs to combine.
        bins (np.ndarray, optional): The bin edges in metres. Defaults to 100 bins between :math:`1\times10^{-7}` and :math:`1\times10^{-7}`.
    """

    def __init__(self, psds: list[PSD], bins: list[float] = None):
        self.psds = psds
        self.psd_weights = [psd.total_number_density for psd in psds]
        self.xlim = (min([psd.xlim[0] for psd in psds]), max([psd.xlim[1] for psd in psds]))
        super().__init__(bins=bins)

    def dn_dd(self, d: ArrayLike) -> np.ndarray:
        """Calculate the particle size distribution value given diameters.
        """
        return sum([psd.dn_dd(d) for psd in self.psds])
    
    def generate_diameters(self, n_particles) -> tuple[list[float], list[ASTModel]]:
        """Generate a particle diameter from the PSD."""
        n_from_psd = [int(n_particles * weight / self.total_number_density) for weight in self.psd_weights]

        #check sum
        if sum(n_from_psd) != n_particles:
            n_from_psd[randint(0,len(self.psds))] += n_particles - sum(n_from_psd)

        diameters = np.array([])
        models = []
        for psd, n in zip(self.psds, n_from_psd):
            psd_diameters, psd_models = psd.generate_diameters(n)
            diameters = np.append(diameters, psd_diameters)
            models += psd_models
        return diameters, models

class GammaPSD(PSD):
    r"""Gamma particle size distribution object.

    Contains the distribution parameters, and binning information.

    .. math::

        n_N(r) = N_0 \left(\frac{r}{\text{1 m}}\right)^\mu \mathrm{e}^{-\Lambda r}.

    Args:
        intercept (float): :math:`N_0` in :math:`\mathrm{m^{-3}}`.
        slope (float): :math:`\Lambda` in :math:`\mathrm{m^{-1}}`.
        shape (float): :math:`\mu`.
        bins (np.ndarray, optional): The bin edges in metres. Defaults to 100 bins between :math:`1\times10^{-7}` and :math:`1\times10^{-7}`.
    """

    def __init__(self, intercept: float, slope: float, shape: float, bins: list[float] = None, model: CrystalModel = CrystalModel.SPHERE):
        self.intercept = intercept
        self.slope = slope
        self.shape = shape

        self.xlim = self.mean + ( 10 * np.sqrt(self.variance) * np.array([-1,1]) )

        super().__init__(bins, model)

    @classmethod
    def from_concentration(cls, number_concentration, slope, shape, **kwargs):
        """Calculate the number density per field.

        Args:
            number_concentration (float): The number concentration in :math:`\mathrm{m^{-3}}`.
            slope (float): :math:`\Lambda` in :math:`\mathrm{m^{-1}}`.
            shape (float): :math:`\mu`.

        Returns:
            float: The number density per field.
        """
        intercept = number_concentration * (slope ** (shape + 1)) / np.math.gamma(shape + 1)
        return cls(intercept, slope, shape, **kwargs)

    @staticmethod
    def _dn_gamma_dd(d:ArrayLike, intercept:float, slope:float, shape:float):
        """The gamma distribution probability distribution function.

        Args:
            d (ArrayLike): The diameters in metres.
            intercept (float): :math:`N_0` in :math:`\mathrm{m^{-4}}`.
            slope (float): :math:`\Lambda` in :math:`\mathrm{m^{-1}}`.
            shape (float): :math:`\mu`.

        Returns:
            callable: The gamma distribution probability distribution function.
        """
        # "shape" expects diameter values in cm in the O'Shea formulation
        return intercept * (d)**shape * np.exp(-1 * slope * d)


    def dn_dd(self, d: ArrayLike) -> np.ndarray:
        """Calculate the particle size distribution value given diameters.
        """
        # "shape" expects diameter values in cm in the O'Shea formulation - hence the 1e-2 fudge factor
        return self._dn_gamma_dd(d, self.intercept, self.slope, self.shape)
    
    @property
    def mean(self):
        """Calculate the mean particle diameter."""
        return (self.shape-1) / self.slope

    @property
    def variance(self):
        """Calculate the variance of the particle diameter."""
        return (self.shape-1) / self.slope**2
    
    @classmethod
    def from_mean_variance(cls, number_concentration, mean, variance, **kwargs):
        """Create a GammaPSD object from the mean and variance of the particle diameter."""
        slope = mean / variance
        shape = (mean * slope) + 1
        return cls.from_concentration(number_concentration, slope, shape, **kwargs)


class OSheaGammaPSD(GammaPSD):
    def dn_dd(self, d: ArrayLike) -> np.ndarray:
        """Calculate the particle size distribution value given diameters.
        """
        # "shape" expects diameter values in cm in the O'Shea formulation - hence the 1e-2 fudge factor
        return self._dn_gamma_dd(d, self.intercept / (1e-2**self.shape), self.slope, self.shape)


class TwoMomentGammaPSD(PSD):

    def __init__(self, m2, m3, bins: list[float] = None, model: CrystalModel = CrystalModel.SPHERE):
        self.m2 = m2
        self.m3 = m3

        super().__init__(bins, model)

    @classmethod
    def from_m2_tc(cls, m2, tc, **kwargs):
        """Create a TwoMomentGammaPSD object from the second moment and in-cloud temperature.
        """
        
        a_n = lambda n: np.exp(13.6 - 7.76 * n + 0.479 * n**2)
        b_n = lambda n: -0.0361 + 0.0151 * n + 0.00149 * n**2
        c_n = lambda n: 0.807 + 0.00581 * n - 0.0457 * n**2

        m3 = a_n(3) * np.exp(b_n(3) * tc) * m2**c_n(3)
        return cls(m2, m3, **kwargs)


    def dn_dd(self, d: ArrayLike) -> np.ndarray:
        """Calculate the particle size distribution value given diameters.
        """
        x = d * self.m2 / self.m3 # now neatly dimensionless
        # TODO: Currently only the mid-latitude parameters
        phi_23 = GammaPSD._dn_gamma_dd(x, 102, 4.82, 2.07)

        return phi_23 / (self.m3**3 / self.m2**4)
    
    @property
    def characteristic_diameter(self):
        return self.m3 / self.m2

class SamplingModel: # NOTE: Not really used.
    """Particle size distribution model.

    A modelling class that contains a PSD object, and simulates particle
    measurement from it. Particles are assumed to be distributed uniformly
    across the inlet. Particle size is assumed to be independent of position
    along the inlet.

    Args:
        psd (PSD): The particle size distribution function, dN/dr in :math:`\mathrm{m^{-3}}`.
        z_dist (callable, optional, unimplimented): The probability 
            distribution of particles along the z-axis, normalised to 1 when 
            integrated wrt z, in m^-1. Defaults to uniform across array.
        inlet_length (float, optional): The length of the inlet in metres.
    """

    def __init__(self, psd: GammaPSD, z_dist: callable = None, inlet_length: float = 7.35e-3):
        self.ast_models = {}
        self.psd = psd

        if z_dist is not None:
            # not implemented
            # TODO: work out xbounds and pmax
            self.z_dist = z_dist
            self.zbounds = [-1, 1]
            self.z_pmax = 1
        else:
            self.z_pmax = 1 / inlet_length
            self.zbounds = [-inlet_length / 2, inlet_length / 2]
            self.z_dist = lambda z: self.z_pmax if abs(
                z) < self.zbounds[1] else 0

    def generate(self, n_particles: int) -> np.ndarray:
        """Generate a sample of particles from the PSD, at random z positions.

        Args:
            n_particles (int): The number of particles to generate.

        Returns:
            np.ndarray: The particles represented by tuples of (diameter, z).
        """
        # generate particle size and z positions
        particles = np.zeros((n_particles, 2))
        for i in range(n_particles):
            particles[i, 0] = choices(
                self.psd.bins[1:], weights=self.psd.binned_distribution
            )[0]
            particles[i, 1] = rejection_sampler(
                self.z_dist, self.zbounds, self.z_pmax)

        return particles

    def simulate_distribution(self, n_particles: int, single_particle: bool = False, keep_models=False) -> np.ndarray:
        """Simulate the observation of spherical particles.

        Generates a sample of spherical particles and then simulates the
        observation of the particles by producing new AST models.

        Args:
            n_particles (int): The number of particles to generate.
            single_particle (bool, optional): Whether to consider only the
                largest particle in each diffraction pattern. Defaults to False.
            keep_models (bool, optional): Whether to keep the AST models after
                simulation. Warning: this will use a lot of memory. Defaults to 
                False.

        Returns:
            np.ndarray: The measured particle diameters.
        """
        particles = self.generate(n_particles)
        diameters_measured = []
        for i in range(particles.shape[0]):
            diameter = particles[i, 0]
            z_value = particles[i, 1]

            if diameter not in self.ast_models:
                self.ast_models[diameter] = ASTModel.from_diameter(
                    diameter / 1e-6)
            intensity = self.ast_models[diameter].process(z_val=z_value).intensity

            if single_particle:
                diameters = [intensity.measure_xy_diameter().tolist()]
            else:
                diameters = intensity.measure_xy_diameters()

            diameters_measured += diameters

            if not keep_models:
                del self.ast_models[diameter]
        return np.array(diameters_measured)

    def simulate_distribution_from_scaling(
            self, n_particles: int, single_particle: bool = False, 
            base_model: ASTModel = None, keep_models=False
            ) -> np.ndarray:
        """Simulate the observation of similar particles.

        Generates a sample of particles, produces new AST models by scaling
        a base model, and then simulates the observation of the particles.

        Args:
            n_particles (int): The number of particles to generate.
            single_particle (bool, optional): Whether to consider only the
                largest particle in each diffraction pattern. Defaults to False.
            base_model (ASTModel, optional): The base model to scale. Should
                have a high resolution relative to typical particle sizes. Defaults
                to a 1000 nm diameter particle.
            keep_models (bool, optional): Whether to keep the AST models after
                simulation. Warning: this will use a lot of memory. Defaults to 
                False.

        Returns:
            np.ndarray: The measured particle diameters.
        """
        if base_model is None:
            base_model = ASTModel.from_diameter(1000)

        base_diameter, _ = base_model.process(0).intensity.measure_xy_diameter()

        particles = self.generate(n_particles)
        diameters_measured = {}
        for i in range(particles.shape[0]):
            diameter = particles[i, 0]
            z_value = particles[i, 1]

            if diameter not in self.ast_models:
                self.ast_models[diameter] = base_model.rescale(
                    (diameter / 1e-6)/base_diameter)
                self.ast_models[diameter].regrid()
            intensity = self.ast_models[diameter].process(z_val=z_value).intensity

            if single_particle:
                diameters = [intensity.measure_xy_diameter().tolist()]
            else:
                diameters = intensity.measure_xy_diameters()

            diameters_measured[(diameter, z_value)] = diameters

            if not keep_models:
                del self.ast_models[diameter]
        # summing with an empty list flattens the list of lists
        return np.array(sum(diameters_measured.values(), [])) 
    
def fit_gamma_distribution(diameters, bins):
    """Fit a gamma distribution to a set of diameters.

    Args:
        diameters (np.ndarray): The diameters to fit.
        bins (np.ndarray): The bins to fit over.

    Returns:
        np.ndarray: The fitted gamma distribution.
    """
    counts, _ = np.histogram(diameters, bins=bins)
    dN_dr = counts / (bins[1:] - bins[:-1]) # TODO: Need to divide by sample volume.... but we don't know what it is.
    bins = bins[:-1]
    bins = bins[counts > 0]
    dN_dr = dN_dr[counts > 0]
    popt, pcov = curve_fit(
        lambda d, intercept, slope: GammaPSD._dn_gamma_dd(d, intercept, slope, 2.5), 
        bins, dN_dr,p0=[1e10, 1e4])
    return popt, pcov