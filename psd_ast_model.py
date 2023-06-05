# Particle size distribution model
# Authour: Oliver Driver
# Date: 01/06/2023

from random import choices
import numpy as np

from ast_model import ASTModel


def rejection_sampler(p, xbounds, pmax):
    """Rejection sampler for a bounded probability distribution."""
    while True:
        x = np.random.rand(1) * (xbounds[1] - xbounds[0]) + xbounds[0]
        y = np.random.rand(1) * pmax
        if y <= p(x):
            return x


class GammaPSD:
    """Gamma particle size distribution $$n_N(r) = N_0 (r/\text{1 m})^\mu e^{-\Gamma r}.$$

    Args:
        intercept (float): N_0 in m^-3.
        slope (float): Gamma in m^-1.
        shape (float): mu.
    """

    def __init__(self, intercept, slope, shape, bins=np.logspace(-7, -3, 100)):
        self.intercept = intercept
        self.slope = slope
        self.shape = shape

        self.bins = bins

    def psd_value(self, r):
        """Calculate the particle size distribution.

        Args:
            r (float): The particle radius in metres.
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
    """Particle size distribution model

    Args:
        ast_model (ASTModel): The AST model to use.
        psd (PSD): The particle size distribution function, dN/dr in m^-1.
        z_dist (callable, optional, unimplimented): The probability distribution of particles along the z-axis, normalised to 1 when integrated wrt z, in m^-1. Defaults to uniform across array.
    """

    def __init__(self, psd, z_dist=None):
        self.ast_models = {}
        self.psd = psd

        if z_dist is not None:
            # not implimented
            # TODO: work out xbounds and pmax
            self.z_dist = z_dist
            self.zbounds = [-1, 1]
            self.z_pmax = 1
        else:
            inlet_length = 7.35e-3  # m
            self.z_pmax = 1 / inlet_length
            self.zbounds = [-inlet_length / 2, inlet_length / 2]
            self.z_dist = lambda z: self.z_pmax if abs(
                z) < self.zbounds[1] else 0

    def generate(self, n_particles):
        """Generate a particle size distribution.

        Args:
            n_particles (int): The number of particles to generate.

        Returns:
            np.ndarray: The particle sizes in metres.
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

    def simulate_distribution(self, n_particles, single_particle=False):
        """ "Generate a measured distribution."""
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
