# Particle size distribution model
# Author: Oliver Driver
# Date: 13/06/2023

from dataclasses import dataclass
from random import choices, seed
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm

from psd_ast_model import GammaPSD
from ast_model import ASTModel, IntensityField, AmplitudeField

# seed(42)
# np.random.seed(42)

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
        total_amplitude = np.zeros((int(self.dimensions[0] / 1e-6), int(self.dimensions[1] / 1e-6)))
        logging.info(f"Calculating intensity profile at z = {z_value} m")
        for particle in tqdm(self.particles.itertuples()):
            ast_model = ASTModel.from_diameter(particle.diameter * 1e6)
            amplitude_at_particle_xy = ast_model.process(particle.position[2] - z_value)

            # embed the amplitude in the total intensity array
            
            total_amplitude = embed_amplitude(amplitude_at_particle_xy, total_amplitude, particle)
        return Slice(z_value, total_amplitude)
    
    def take_image(self,  detector_position: np.ndarray, arm_separation: float=10e-2, distance:float=10e-6):
        """Take an image using repeated detections along the y-axis.

        Detector is aligned with x-axis, and the y-axis is the direction of travel.
        The z-axis is the focal axis.
        The detector position is the position of the detector centre during the final detection.
        
        """

        n_images = int(distance / 10e-6)

        # check which particles are somewhat within within the illuminated region
        # Illuminated region is defined as the 8mm x 2mm x arm_separation region aligned with the orthogonal vector pointing towards the detector
        # TODO: These need to also detect particles whose centres are outside the illuminated region, but whos edges are inside it.
        is_in_illuminated_region_x = lambda particle: abs(np.dot(particle.position - detector_position, np.array([1,0,0]))) < 4e-3
        is_in_illuminated_region_y = lambda particle: np.dot(particle.position - detector_position, np.array([0,1,0])) < 1e-3 and np.dot(particle.position - detector_position, np.array([0,1,0])) > (-1e-3 - n_images * 10e-6)
        is_in_illuminated_region_z = lambda particle: np.dot(particle.position - detector_position, np.array([0,0,1])) < arm_separation and np.dot(particle.position - detector_position, np.array([0,0,1])) > 0 

        in_illuminated_region = self.particles.apply(
            lambda particle: is_in_illuminated_region_x(particle) and is_in_illuminated_region_y(particle) and is_in_illuminated_region_z(particle),
            axis=1
        )
        if not in_illuminated_region.any():
            # No particles are in the illuminated region.
            return None

        # get the intensity profile at the given z value for each illuminated particle
        total_amplitude = AmplitudeField(np.ones((128,n_images), dtype=np.complex128), pixel_size=10e-6)
        for particle in self.particles[in_illuminated_region].itertuples():
            ast_model = ASTModel.from_diameter(particle.diameter * 1e6)
            amplitude_at_particle_xy = ast_model.process(particle.position[2] - detector_position[2] - arm_separation/2)

            total_amplitude = embed_amplitude(amplitude_at_particle_xy, total_amplitude, particle, detector_position)

        return ImagedRegion(detector_position, total_amplitude)


def embed_amplitude(single_particle_amplitude, total_amplitude, particle, detector_position):
    """Embed the intensity profile of a particle into the total intensity array."""

    # vector from particle to detector
    pcle_from_detector = particle.position - detector_position

    
    # index of particle centre in total_intensity
    # detector is at x = total_amplitude.shape[0]/2, y = 0
    x_index = int(pcle_from_detector[0] / total_amplitude.pixel_size) + int(total_amplitude.shape[0]/2) 
    y_index = int(pcle_from_detector[1] / total_amplitude.pixel_size)

    amplitude_shape = single_particle_amplitude.shape

    # Check pixel sizes are consistent
    if single_particle_amplitude.pixel_size != total_amplitude.pixel_size:
        raise ValueError(f"Pixel sizes of single_particle_amplitude and total_amplitude must be the same.\nSingle particle: {single_particle_amplitude.pixel_size} m, Total: {total_amplitude.pixel_size} m")

    # determine the bounds of the total intensity array to embed the particle intensity in
    # "do it to the edge, but not over the edge"
    if x_index < int(amplitude_shape[0]/2):
        # would be out of bounds at x=0
        # go to edge of total_intensity and trim single_particle_intensity
        total_x_min = 0
        single_x_min = int(amplitude_shape[0]/2) - x_index
    else:
        total_x_min = x_index - int(amplitude_shape[0]/2)
        single_x_min = 0
    
    if y_index < int(amplitude_shape[1]/2):
        # would be out of bounds at y=0
        total_y_min = 0
        single_y_min = int(amplitude_shape[1]/2) - y_index
    else:
        total_y_min = y_index - int(amplitude_shape[1]/2)
        single_y_min = 0
    
    if x_index - int(amplitude_shape[0]/2) + amplitude_shape[0] > total_amplitude.shape[0]:
        # would be out of bounds at max x
        total_x_max = total_amplitude.shape[0]
        # single_size - ((endpoint) - total_size)
        single_x_max = amplitude_shape[0] - ((x_index - int(amplitude_shape[0]/2) + amplitude_shape[0]) - total_amplitude.shape[0])
    else:
        total_x_max = x_index - int(amplitude_shape[0]/2) + amplitude_shape[0]
        single_x_max = amplitude_shape[0]

    if y_index - int(amplitude_shape[1]/2) + amplitude_shape[1] > total_amplitude.shape[1]:
        # would be out of bounds at max y
        total_y_max = total_amplitude.shape[1]
        single_y_max = amplitude_shape[1] - ((y_index - int(amplitude_shape[1]/2) + amplitude_shape[1]) - total_amplitude.shape[1])
    else:
        total_y_max = y_index - int(amplitude_shape[1]/2) + amplitude_shape[1]
        single_y_max = amplitude_shape[1]

    # check for the non-overlapping case
    if total_x_min > total_amplitude.shape[0] or total_y_min > total_amplitude.shape[1] or total_x_max < 0 or total_y_max < 0:
        return total_amplitude

    #TODO: check this....... Are the amplitudes combined correctly
    new_amplitude = single_particle_amplitude[single_x_min:single_x_max, single_y_min:single_y_max]
    total_amplitude[total_x_min:total_x_max, total_y_min:total_y_max] *= new_amplitude

    # new_amplitude_reshaped = AmplitudeField(np.ones_like(total_amplitude, dtype=np.complex128), pixel_size=10e-6)
    # new_amplitude_reshaped[total_x_min:total_x_max, total_y_min:total_y_max] = new_amplitude

    # total_amplitude = AmplitudeField(np.fft.ifft2(total_amplitude.phase * new_amplitude_reshaped.phase), pixel_size=10e-6)
    # total_amplitude[total_x_min:total_x_max, total_y_min:total_y_max] *= single_particle_amplitude[single_x_min:single_x_max, single_y_min:single_y_max] - 1
    return total_amplitude

@dataclass
class Slice:
    z_value: float # in m
    amplitude: AmplitudeField # relative intensity

@dataclass
class ImagedRegion:
    """A region illuminated by the laser beam at a single instant."""
    # orthogonal_vector: np.ndarray = (0,0,1) # in m
    detector_position: np.ndarray # in m
    amplitude: AmplitudeField # relative intensity
    arm_separation: float = 10e-2# in m

@dataclass
class Particle:
    diameter: float # in m
    position: tuple[float, float, float] # (x, y, z) in m