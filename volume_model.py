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
    

@dataclass
class Particle:
    diameter: float # in m
    position: tuple[float, float, float] # (x, y, z) in m