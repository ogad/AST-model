# Particle size distribution model
# Author: Oliver Driver
# Date: 13/06/2023

from dataclasses import dataclass
from random import choices
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm

from psd_ast_model import GammaPSD
from ast_model import ASTModel

@dataclass
class CloudVolume:
    psd: GammaPSD
    dimensions: tuple[float, float, float] # (x, y, z) in m

    @staticmethod
    def _generate_position(dim_grids):
        """Generate a random position within the cloud volume."""
        position =  np.array([np.random.choice(dim_grid) for dim_grid in dim_grids])
        return position
    
    def __post_init__(self):
        logging.info("Initialising cloud volume")
        
        logging.info(f"Generating grid of dimensions: {[int(dim / 1e-6) for dim in self.dimensions]} points.")
        # raise warning if any dimension will have more than 2e9 points
        if any([dim > 2e3 for dim in self.dimensions]):
            logging.warn("One or more dimensions is too great to grid at 2µm resolution.")
        dim_grids = [np.arange(0, dim, 2e-6) for dim in self.dimensions]

        # Generate the particles
        self.particles = pd.DataFrame(columns=["diameter", "position"])
        logging.info(f"Generating {self.n_particles} particles")

        for i in tqdm(range(self.n_particles)):
            particle = [self.psd.generate_diameter(), self._generate_position(dim_grids)]
            self.particles.loc[i] = particle

    @property
    def n_particles(self):
        """Calculate the number of particles in the volume based on the PSD."""
        return int(self.volume * self.psd.total_number_density)

    @property
    def volume(self):
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    
    def slice(self, z_value: float):
        """Return a slice of the cloud volume at a given z value."""
        if z_value < 0 or z_value > self.dimensions[2]:
            raise ValueError("z value must be within the cloud volume.")
        
        # get the intensity profile at the given z value for each particle
        total_intensity = np.zeros((int(self.dimensions[0] / 1e-6), int(self.dimensions[1] / 1e-6)))
        logging.info(f"Calculating intensity profile at z = {z_value} m")
        for particle in tqdm(self.particles.itertuples()):
            ast_model = ASTModel.from_diameter(particle.diameter * 1e6)
            intensity_at_particle_xy = ast_model.process(particle.position[2] - z_value)

            # embed the intensity in the total intensity array
            x_index = int(particle.position[0] / 1e-6)
            y_index = int(particle.position[1] / 1e-6)
            intensity_shape = intensity_at_particle_xy.shape
            x_min = max(0, x_index - int(intensity_shape[0]/2))
            x_max = min(total_intensity.shape[0], x_index - int(intensity_shape[0]/2) + intensity_shape[0])
            y_min = max(0, y_index - int(intensity_shape[1]/2))
            y_max = min(total_intensity.shape[1], y_index - int(intensity_shape[1]/2) + intensity_shape[1])
            
            intensity_to_embed = intensity_at_particle_xy[
                max(0, int(intensity_shape[0]/2) - x_index):min(intensity_shape[0], total_intensity.shape[0] - x_index + int(intensity_shape[0]/2)),
                max(0, int(intensity_shape[1]/2) - y_index):min(intensity_shape[1], total_intensity.shape[1] - y_index + int(intensity_shape[1]/2))
            ]

            total_intensity[x_min:x_max, y_min:y_max] += intensity_to_embed
        return Slice(z_value, total_intensity)



@dataclass
class Slice:
    z_value: float # in m
    intensity: np.ndarray # in W/m^2

@dataclass
class Particle:
    diameter: float # in m
    position: tuple[float, float, float] # (x, y, z) in m