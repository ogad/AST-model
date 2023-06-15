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
from ast_model import ASTModel, IntensityField

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
        # FIXME: this doesnt work. Intensities add but with a default of 1, so you get overlapping squares.
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
            
            total_intensity = embed_intensity(intensity_at_particle_xy, total_intensity, particle)
        return Slice(z_value, total_intensity)
    
    def take_image(self,  detector_position: np.ndarray, arm_separation: float=10e-2):
        """Take a single array measurement."""

        # check which particles are somewhat within within the illuminated region
        # Illuminated region is defined as the 8mm x 2mm x arm_separation region aligned with the orthogonal vector pointing towards the detector
        # TODO: These need to also detect particles whose centres are outside the illuminated region, but whos edges are inside it.
        is_in_illuminated_region_x = lambda particle: np.dot(particle.position - detector_position, np.array([1,0,0])) < 4e-3 and np.dot(particle.position - detector_position, np.array([1,0,0])) > -4e-3 
        is_in_illuminated_region_y = lambda particle: np.dot(particle.position - detector_position, np.array([0,1,0])) < 1e-3 and np.dot(particle.position - detector_position, np.array([0,1,0])) > -1e-3
        is_in_illuminated_region_z = lambda particle: np.dot(particle.position - detector_position, np.array([0,0,1])) < arm_separation and np.dot(particle.position - detector_position, np.array([0,0,1])) > 0 

        in_illuminated_region = self.particles.apply(
            lambda particle: is_in_illuminated_region_x(particle) and is_in_illuminated_region_y(particle) and is_in_illuminated_region_z(particle),
            axis=1
        )

        # get the intensity profile at the given z value for each illuminated particle
        total_intensity = IntensityField(np.ones((128,1)), pixel_size=10e-6)
        for particle in self.particles[in_illuminated_region].itertuples():
            ast_model = ASTModel.from_diameter(particle.diameter * 1e6)
            intensity_at_particle_xy = ast_model.process(particle.position[2] - detector_position[2] - arm_separation/2)

            total_intensity = embed_intensity(intensity_at_particle_xy, total_intensity, particle, detector_position)

        return ImagedRegion(detector_position, total_intensity)


def embed_intensity(single_particle_intensity, total_intensity, particle, detector_position):
    """Embed the intensity profile of a particle into the total intensity array."""
    # vector from particle to detector
    pcle_from_detector = particle.position - detector_position
    # index of particle centre in total_intensity
    x_index = int(pcle_from_detector[0] / total_intensity.pixel_size) + int(total_intensity.shape[0]/2)
    y_index = int(pcle_from_detector[1] / total_intensity.pixel_size) + int(total_intensity.shape[1]/2)

    intensity_shape = single_particle_intensity.shape

    # Check pixel sizes are consistent
    if single_particle_intensity.pixel_size != total_intensity.pixel_size:
        raise ValueError(f"Pixel sizes of single_particle_intensity and total_intensity must be the same.\nSingle particle: {single_particle_intensity.pixel_size} m, Total: {total_intensity.pixel_size} m")

    # determine the bounds of the total intensity array to embed the particle intensity in
    # "do it to the edge, but not over the edge"
    if x_index < int(intensity_shape[0]/2):
        # would be out of bounds at x=0
        # go to edge of total_intensity and trim single_particle_intensity
        total_x_min = 0
        single_x_min = int(intensity_shape[0]/2) - x_index
    else:
        total_x_min = x_index - int(intensity_shape[0]/2)
        single_x_min = 0
    
    if y_index < int(intensity_shape[1]/2):
        # would be out of bounds at y=0
        total_y_min = 0
        single_y_min = int(intensity_shape[1]/2) - y_index
    else:
        total_y_min = y_index - int(intensity_shape[1]/2)
        single_y_min = 0
    
    if x_index - int(intensity_shape[0]/2) + intensity_shape[0] > total_intensity.shape[0]:
        # would be out of bounds at max x
        total_x_max = total_intensity.shape[0]
        # single_size - ((endpoint) - total_size)
        single_x_max = intensity_shape[0] - ((x_index - int(intensity_shape[0]/2) + intensity_shape[0]) - total_intensity.shape[0])
    else:
        total_x_max = x_index - int(intensity_shape[0]/2) + intensity_shape[0]
        single_x_max = intensity_shape[0]

    if y_index - int(intensity_shape[1]/2) + intensity_shape[1] > total_intensity.shape[1]:
        # would be out of bounds at max y
        total_y_max = total_intensity.shape[1]
        single_y_max = intensity_shape[1] - ((y_index - int(intensity_shape[1]/2) + intensity_shape[1]) - total_intensity.shape[1])
    else:
        total_y_max = y_index - int(intensity_shape[1]/2) + intensity_shape[1]
        single_y_max = intensity_shape[1]

    total_intensity[total_x_min:total_x_max, total_y_min:total_y_max] += single_particle_intensity[single_x_min:single_x_max, single_y_min:single_y_max] -1
    return total_intensity






@dataclass
class Slice:
    z_value: float # in m
    intensity: np.ndarray # relative intensity

@dataclass
class ImagedRegion:
    """A region illuminated by the laser beam at a single instant."""
    # orthogonal_vector: np.ndarray = (0,0,1) # in m
    detector_position: np.ndarray # in m
    intensity: np.ndarray # relative intensity
    arm_separation: float = 10e-2# in m

@dataclass
class Particle:
    diameter: float # in m
    position: tuple[float, float, float] # (x, y, z) in m