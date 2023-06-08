# Particle size distribution model
# Authour: Oliver Driver
# Date: 01/06/2023

from random import choices
import numpy as np
from numpy.typing import ArrayLike


from ast_model import ASTModel


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


class GammaPSD:
    r"""Gamma particle size distribution object.

    Contains the distribution parameters, and binning information.

    .. math::

        n_N(r) = N_0 (r/\text{1 m})^\mu e^{-\Lambda r}.

    Args:
        intercept (float): :math:`N_0` in m^-3.
        slope (float): :math:`\Lambda` in m^-1.
        shape (float): :math:`\mu`.
        bins (np.ndarray, optional): The bin edges in metres. Defaults to 100 bins between 1e-7 and 1e-3.
    """

    def __init__(self, intercept: float, slope: float, shape: float, bins: ArrayLike[np.float64] = None):
        self.intercept = intercept
        self.slope = slope
        self.shape = shape

        if bins is None:
            self.bins = np.logspace(-7, -3, 100)
        else:
            self.bins = bins

    def psd_value(self, r: ArrayLike[np.float64]):
        """Calculate the particle size distribution value given radii.
        """
        return self.intercept * r**self.shape * np.exp(-1 * self.slope * r)

    @property
    def binned_distribution(self):
        """Calculate the binned particle size distribution.

        Returns:
            np.ndarray: The number of particles in each bin.
        """
        return self.psd_value(self.bins[1:]) * np.diff(self.bins)


class PSDModel:
    """Particle size distribution model.

    A modelling class that contains a PSD object, and simulates particle
    measurement from it. Particles are assumed to be distributed uniformly
    across the inlet. Particle size is assumed to be independent of position
    along the inlet.

    Args:
        psd (PSD): The particle size distribution function, dN/dr in m^-1.
        z_dist (callable, optional, unimplimented): The probability 
            distribution of particles along the z-axis, normalised to 1 when 
            integrated wrt z, in m^-1. Defaults to uniform across array.
        inlet_length (float, optional): The length of the inlet in metres.
    """

    def __init__(self, psd: GammaPSD, z_dist: callable = None, inlet_length: float = 7.35e-3):
        self.ast_models = {}
        self.psd = psd

        if z_dist is not None:
            # not implimented
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
        """Generate a particle size distribution.

        Args:
            n_particles (int): The number of particles to generate.

        Returns:
            np.ndarray: The particles represented by tuples of (radius, z).
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

    def simulate_distribution(self, n_particles: int, single_particle: bool = False) -> np.ndarray:
        """Simulate the observation of spherical particles.

        Generates a sample of spherical particles and then simulates the
        observation of the particles by producing new AST models.

        Args:
            n_particles (int): The number of particles to generate.
            single_particle (bool, optional): Whether to consider only the
                largest particle in each diffraction pattern. Defaults to False.

        Returns:
            np.ndarray: The measured particle diameters.
        """
        particles = self.generate(n_particles)
        diameters_measured = []
        for i in range(particles.shape[0]):
            radius = particles[i, 0]
            z_value = particles[i, 1]

            if radius not in self.ast_models:
                self.ast_models[radius] = ASTModel.from_diameter(
                    radius * 2 / 1e-6)
            intensity = self.ast_models[radius].process(z_val=z_value)

            if single_particle:
                diameters = [intensity.measure_xy_diameter().tolist()]
            else:
                diameters = intensity.measure_xy_diameters()

            diameters_measured += diameters
        return np.array(diameters_measured)

    def simulate_distribution_from_scaling(self, n_particles: int, single_particle: bool = False, base_model: ASTModel = None):
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

        Returns:
            np.ndarray: The measured particle diameters.
        """
        if base_model is None:
            base_model = ASTModel.from_diameter(1000)

        base_diameter = base_model.process(0).measure_xy_diameter()

        particles = self.generate(n_particles)
        diameters_measured = {}
        for i in range(particles.shape[0]):
            radius = particles[i, 0]
            z_value = particles[i, 1]

            if radius not in self.ast_models:
                self.ast_models[radius] = base_model.rescale(
                    (radius * 2 / 1e-6)/base_diameter)
                self.ast_models[radius].regrid()
            intensity = self.ast_models[radius].process(z_val=z_value)

            if single_particle:
                diameters = [intensity.measure_xy_diameter().tolist()]
            else:
                diameters = intensity.measure_xy_diameters()

            diameters_measured[(radius, z_value)] = diameters
        return np.array(sum(diameters_measured.values(), []))
