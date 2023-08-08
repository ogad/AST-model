# Particle size distribution model
# Author: Oliver Driver
# Date: 01/06/2023

from collections import namedtuple
from random import choices
from abc import ABC, abstractmethod
from enum import Enum
import logging

import numpy as np
from numpy.random import randint
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit, root_scalar


from ast_model import ASTModel
from diameters import measure_diameters


Particle = namedtuple("Particle", ["diameter", "angle", "model"])
PositionedParticle = namedtuple("PositionedParticle", ["diameter",  "angle", "model", "position"])

class CrystalModel(Enum):
    """Enum for crystal types."""
    SPHERE = 1
    RECT_AR5 = 2
    ROS_6 = 3
    COL_AR5_ROT = 4
    RECT_AR5_ROT = 5

    def __init__(self, *args) -> None:
        self.model_names = {
            1: "Sphere",
            2: "Rectangular with aspect ratio 5",
            3: "Rosette with 6 arms, 100µm width",
            4: "Rectangular with aspect ratio 5, rotated in 3D",
            5: "Rectangular with aspect ratio 5, rotated in plane"
        }
        super().__init__(*args)

    def __str__(self):
        return self.model_names[self.value]

    def get_generator(self):
        if self == CrystalModel.SPHERE:
            return lambda particle, **kwargs: ASTModel.from_diameter(particle.diameter*1e6, **kwargs)
        elif self == CrystalModel.RECT_AR5:
            return lambda particle, **kwargs: ASTModel.from_diameter_rectangular(particle.diameter*1e6, 5, **kwargs)
        elif self == CrystalModel.RECT_AR5_ROT:
            return lambda particle, **kwargs: ASTModel.from_diameter_rectangular(particle.diameter*1e6, 5, angle=particle.angle[0], **kwargs)
        elif self == CrystalModel.COL_AR5_ROT:
            return lambda particle, **kwargs: ASTModel.from_diameter_rectangular(particle.diameter*1e6, 5, angle=particle.angle, **kwargs)
        elif self == CrystalModel.ROS_6:
            return lambda particle, **kwargs: ASTModel.from_diameter_rosette(particle.diameter*1e6, 3, **kwargs)
        else:
            raise ValueError("Crystal model not recognised.")
        
    def min_diameter(self, pixel_size):
        model_generator = self.get_generator()
        
        diameter = 0
        while True:
            model = model_generator(Particle(diameter,(0,0),self), pixel_size=pixel_size)
            if model.opaque_shape.any():
                break
            diameter += pixel_size/10
        
        return diameter




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
            lower_lim = -7 if self.xlim[0] == 0 else np.log10(self.xlim[0])
            bins = np.logspace(lower_lim, np.log10(self.xlim[1]), 1000)
        elif bins is None:
            bins = np.logspace(-7, -3, 1000)
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
        return self.dn_dd(self.midpoints) * (np.diff(self.bins))
    
    @property
    def midpoints(self):
        """Calculate the binned particle size distribution.

        Returns:
            np.ndarray: The number of particles in each bin.
        """
        midpoints = (self.bins[1:] + self.bins[:-1]) / 2
        return midpoints
    
    @property
    def max_dn_dd(self):
        """Calculate the maximum value of the particle size distribution."""
        return self.dn_dd(self.midpoints).max()
    
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
    
    def plot(self, ax, retrieval:'Retrieval'=None, **kwargs):
        """Plot the PSD value against diameter."""
        if retrieval is None: # Don't adjust
            x_vals = self.bins
        else:
            x_vals = self.adjusted_bins(retrieval)
        handle = ax.plot(x_vals, self.dn_dd(self.bins), **kwargs)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel('Diameter (m)')
        ax.set_ylabel('PSD ($\mathrm{m}^{-3}\,\mathrm{m}^{-1}$)')
        return handle
    
    def adjusted_bins(self, retrieval):
        # for each bin edge, calculate the focused diameter
        adjusted_bins = []
        
        for diameter in self.bins:
            particle = Particle(diameter, (0,0), self.model)
            model = self.model.get_generator()(particle,  wavenumber=2*np.pi/retrieval.run.detector.wavelength, pixel_size=retrieval.run.detector.pixel_size)
            focused_amp = model.process(0)
            diameter = measure_diameters(focused_amp, retrieval.spec, force_nominsep=True)
            if len(diameter) > 1:
                raise ValueError("Multiple diameters found in focused image... Something is very wrong.")
            elif len(diameter) == 0:
                adjusted_bins.append(0)
            else:
                adjusted_bins.append(next(iter(diameter.values())))
        return np.array(adjusted_bins) * 1e-6

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
        self.xlim[0] = max(self.xlim[0], 0)
        self.xlim[1] = max(self.xlim[1], 5e-4)

        super().__init__(bins, model)

    def __str__(self) -> str:
        return super().__str__() + f"GammaPSD({self.intercept}, {self.slope}, {self.shape})"

    @classmethod
    def from_litres_microns(cls, intercept_per_litre_per_micron, slope_per_micron, shape, **kwargs):
        # Note the extra factor of 1e-6^shape for conversion of power law.
        intercept = intercept_per_litre_per_micron * 1e3 * 1e6 * (1e6**shape)
        slope = slope_per_micron * 1e6
        return cls(intercept, slope, shape, **kwargs)
    
    @classmethod
    def from_litres_cm(cls, intercept_per_litre_per_cm, slope_per_cm, shape, **kwargs):
        # Note the extra factor of 1e-2^shape for conversion of power law.
        intercept = intercept_per_litre_per_cm * 1e3 * (1e2**shape)
        slope = slope_per_cm * 1e2
        return cls(intercept, slope, shape, **kwargs)
    
    @classmethod
    def w19_parameterisation(cls, temp, intercept, insitu_origin=False, liquid_origin=False):
        """Using Wolf et al. 2019 parameterisation.


        Returns:
            float: The number density per field.
        """
        if not (insitu_origin or liquid_origin):
            raise ValueError("Must specify either insitu_origin or liquid_origin.")
        if insitu_origin and liquid_origin:
            raise ValueError("Cannot specify both insitu_origin and liquid_origin.")
        
        if insitu_origin:
            ln_slope_param = lambda temp: -0.06837 * temp + 3.492 #cm^-1
            shape_param = lambda ln_slope: 0.009*np.exp(ln_slope*0.85) # Muhlbauer 2014
            # shape_param = lambda ln_slope: 0.02819 * np.exp(0.7216*ln_slope) 
        elif liquid_origin:
            ln_slope_param = lambda temp: 4.937 * np.exp(-0.001846*temp) #cm^-1
            shape_param = lambda ln_slope: 0.104*np.exp(ln_slope*0.71) - 1.7
            # shape_param = lambda ln_slope: 0.001379 * np.exp(1.285*ln_slope)

        ln_slope = ln_slope_param(temp) 
        shape = shape_param(ln_slope)


        return cls.from_litres_cm(intercept, np.exp(ln_slope), shape)

    def parameter_description(self) -> str:
        return f"$N_0={self.intercept:.2e}$, $\lambda={self.slope:.2e}$, $\mu={self.shape:.2f}$"

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
        return np.abs((self.shape-1) / self.slope**2)
    
    @classmethod
    def from_mean_variance(cls, number_concentration, mean, variance, **kwargs):
        """Create a GammaPSD object from the mean and variance of the particle diameter."""
        slope = mean / variance
        shape = (mean * slope) + 1
        return cls.from_concentration(number_concentration, slope, shape, **kwargs)
    
    @classmethod
    def fit(cls, diameters, dn_dd, min_considered_diameter=50e-6):
        from scipy.optimize import curve_fit

        diameter_vals = diameters[(diameters >= min_considered_diameter)]
        dn_dd_vals = dn_dd[(diameters >= min_considered_diameter)]

        # expon = lambda d, intercept, slope: intercept * np.exp(-1 * slope * d)

        # log_gamma = lambda d, intercept, slope, shape: np.log10(GammaPSD._dn_gamma_dd(d, intercept, slope, shape))

        fit_gamma = lambda d, intercept, slope, shape: GammaPSD._dn_gamma_dd(d, intercept, slope, shape)

        results = curve_fit(GammaPSD._dn_gamma_dd, diameter_vals, dn_dd_vals / 1e6, 
                                p0=[1, 1, 1], 
                                maxfev=10000,
                                bounds=([0, 0, 1], [np.inf, np.inf, np.inf]),
                                # method='dogbox',
                                full_output=True
                               ) 
        
        intercept_l_mcb, slope, shape = results[0]
        intercept = intercept_l_mcb * 1e6

        logging.info(f"\t{results[3]}")

        return cls(intercept, slope, shape)



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

class SamplingModel: # NOTE: Not really used anymore; deprecated in favour of CloudModel, DetectorRun, and Retrieval
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
    
# def fit_gamma_distribution(diameters, bins): # NOTE: Not really used anymore
#     """Fit a gamma distribution to a set of diameters.

#     Args:
#         diameters (np.ndarray): The diameters to fit.
#         bins (np.ndarray): The bins to fit over.

#     Returns:
#         np.ndarray: The fitted gamma distribution.
#     """
#     counts, _ = np.histogram(diameters, bins=bins)
#     dN_dr = counts / (bins[1:] - bins[:-1]) # TODO: Need to divide by sample volume.
#     bins = bins[:-1]
#     bins = bins[counts > 0]
#     dN_dr = dN_dr[counts > 0]
#     popt, pcov = curve_fit(
#         lambda d, intercept, slope: GammaPSD._dn_gamma_dd(d, intercept, slope, 2.5), 
#         bins, dN_dr,p0=[1e10, 1e4])
#     return popt, pcov