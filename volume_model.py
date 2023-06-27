# Particle size distribution model
# Author: Oliver Driver
# Date: 13/06/2023

from dataclasses import dataclass
from random import choices, seed
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from enum import Enum

import matplotlib.pyplot as plt

from psd_ast_model import GammaPSD
from ast_model import ASTModel, IntensityField, AmplitudeField, plot_outline

# seed(42)
# np.random.seed(42)


@dataclass
class Detector:
    position: tuple[float, float, float] # (x, y, z) in m
    arm_separation: float = 10e-2 # in m
    n_pixels: int = 128
    pixel_size: float =10e-6# in m

class CrystalModel(Enum):
    """Enum for crystal types."""
    SPHERE = 1
    RECT_AR5 = 2

    def get_generator(self):
        if self == CrystalModel.SPHERE:
            return ASTModel.from_diameter
        elif self == CrystalModel.RECT_AR5:
            return lambda diameter: ASTModel.from_diameter_rectangular(diameter, 5)
        else:
            raise ValueError("Crystal model not recognised.")
    
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
        self.particles = pd.DataFrame(columns=["diameter", "position", "model"])
        logging.info(f"Generating {self.n_particles} particles")

        for i in tqdm(range(self.n_particles), total=self.n_particles):
            particle = [self.psd.generate_diameter(), self._generate_position(dim_grids), CrystalModel.SPHERE]
            self.particles.loc[i] = particle

    @property
    def n_particles(self):
        """Calculate the number of particles in the volume based on the PSD."""
        return int(self.volume * self.psd.total_number_density)

    @property
    def volume(self):
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    
    def take_image(self, detector: Detector, distance:float=10e-6, offset: np.ndarray = np.array([0,0,0])  ,separate_particles: bool=False, use_focus: bool=False, primary_only: bool=False):
        """Take an image using repeated detections along the y-axis.

        Detector is aligned with x-axis, and the y-axis is the direction of travel.
        The z-axis is the focal axis.
        The detector position is the position of the detector centre during the final detection.
        The model_generator is a callable taking the diameter of the particle in microns and returning an ASTModel.
        
        """

        detector_position = detector.position + offset

        n_images = int(distance / detector.pixel_size)
        beam_width = 8e-3 - (128 * 10e-6) + (detector.n_pixels * detector.pixel_size) # leave the same padding
        beam_length = 2e-3 - (10e-6) + (detector.pixel_size) # leave the same padding

        # check which particles are somewhat within within the illuminated region
        # Illuminated region is defined as the 8mm x 2mm x arm_separation region aligned with the orthogonal vector pointing towards the detector
        # TODO: These need to also detect particles whose centres are outside the illuminated region, but whos edges are inside it.
        is_in_illuminated_region_x = lambda particle: abs(np.dot(particle.position - detector_position, np.array([1,0,0]))) < (beam_width/2)
        is_in_illuminated_region_y = lambda particle: np.dot(particle.position - detector_position, np.array([0,1,0])) < (beam_length/2 + n_images * detector.pixel_size) and np.dot(particle.position - detector_position, np.array([0,1,0])) > -1*beam_length/2
        is_in_illuminated_region_z = lambda particle: np.dot(particle.position - detector_position, np.array([0,0,1])) < detector.arm_separation and np.dot(particle.position - detector_position, np.array([0,0,1])) > 0 

        in_illuminated_region = self.particles.apply(
            lambda particle: is_in_illuminated_region_x(particle) and is_in_illuminated_region_y(particle) and is_in_illuminated_region_z(particle),
            axis=1
        )
        if not in_illuminated_region.any():
            # No particles are in the illuminated region.
            return None

        if separate_particles:
            # take an image of each particle individually
            images = []
            length_iterations = len(self.particles[in_illuminated_region])
            for particle in tqdm(self.particles[in_illuminated_region].itertuples(), total=length_iterations, leave=False):
                if primary_only and not particle.primary:
                    continue
                y_offset = particle.position[1] - detector_position[1] - 5 * particle.diameter
                z_offset = particle.position[2] - (detector_position[2] + detector.arm_separation/2) if use_focus else 0
                particle_image = self.take_image(detector, 10*particle.diameter, offset=np.array([0, y_offset, z_offset]), use_focus=use_focus)
                particle_image.particles["primary"] = particle_image.particles.index == particle.Index

                images.append(particle_image)

            particles = self.particles[in_illuminated_region].copy()
            return images, particles
        


        # get the intensity profile at the given z value for each illuminated particle
        total_amplitude = AmplitudeField(np.ones((detector.n_pixels, n_images), dtype=np.complex128), pixel_size=detector.pixel_size)
        
        particles = self.particles[in_illuminated_region].copy()
        particles["x_index"] = particles.apply(
            lambda particle: int((particle["position"] - detector_position)[0] / total_amplitude.pixel_size) + int(total_amplitude.shape[0]/2),
            axis=1)
        particles["y_index"] = particles.apply(
            lambda particle: int((particle["position"] - detector_position)[1] / total_amplitude.pixel_size),
            axis=1)

        for particle in particles.itertuples():
            model_generator = particle.model.get_generator() if particle.model is not None else model_generator
            ast_model = model_generator(particle.diameter * 1e6)
            amplitude_at_particle_xy = ast_model.process(particle.position[2] - detector_position[2] - detector.arm_separation/2)

            total_amplitude = embed_amplitude(amplitude_at_particle_xy, total_amplitude, particle, detector_position)
        
        image = ImagedRegion(detector_position, total_amplitude, particles=particles)
        return image


def embed_amplitude(single_particle_amplitude, total_amplitude, particle, detector_position):
    """Embed the intensity profile of a particle into the total intensity array."""

    # vector from particle to detector
    pcle_from_detector = particle.position - detector_position

    
    # index of particle centre in total_intensity
    # detector is at x = total_amplitude.shape[0]/2, y = 0
    x_index = int(pcle_from_detector[0] / total_amplitude.pixel_size + total_amplitude.shape[0]/2)
    y_index = total_amplitude.shape[1] - int(pcle_from_detector[1] / total_amplitude.pixel_size)
    embed_extent = [
        x_index - int(single_particle_amplitude.shape[0]/2), 
        x_index - int(single_particle_amplitude.shape[0]/2) + single_particle_amplitude.shape[0], 
        y_index - int(single_particle_amplitude.shape[1]/2), 
        y_index - int(single_particle_amplitude.shape[1]/2) + single_particle_amplitude.shape[1]
    ]

    amplitude_shape = single_particle_amplitude.shape

    # Check pixel sizes are consistent
    if single_particle_amplitude.pixel_size != total_amplitude.pixel_size:
        raise ValueError(f"Pixel sizes of single_particle_amplitude and total_amplitude must be the same.\nSingle particle: {single_particle_amplitude.pixel_size} m, Total: {total_amplitude.pixel_size} m")

    # determine the bounds of the total intensity array to embed the particle intensity in
    # "do it to the edge, but not over the edge"
    if embed_extent[0] < 0:
        # would be out of bounds at x=0
        # go to edge of total_intensity and trim single_particle_intensity
        total_x_min = 0
        single_x_min = int(amplitude_shape[0]/2) - x_index
    else:
        total_x_min = embed_extent[0]
        single_x_min = 0
    
    if embed_extent[2] < 0:
        # would be out of bounds at y=0
        total_y_min = 0
        single_y_min = int(amplitude_shape[1]/2) - y_index
    else:
        total_y_min = embed_extent[2]
        single_y_min = 0
    
    if embed_extent[1] > total_amplitude.shape[0]:
        # would be out of bounds at max x
        total_x_max = total_amplitude.shape[0]
        # single_size - ((endpoint) - total_size)
        single_x_max = amplitude_shape[0] - (embed_extent[1] - total_amplitude.shape[0])
    else:
        total_x_max = embed_extent[1]
        single_x_max = amplitude_shape[0]

    if embed_extent[3] > total_amplitude.shape[1]:
        # would be out of bounds at max y
        total_y_max = total_amplitude.shape[1]
        single_y_max = amplitude_shape[1] - (embed_extent[3] - total_amplitude.shape[1])
    else:
        total_y_max = embed_extent[3]
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
class ImagedRegion:
    """A region illuminated by the laser beam at a single instant."""
    # orthogonal_vector: np.ndarray = (0,0,1) # in m
    detector_position: np.ndarray # in m
    amplitude: AmplitudeField # relative intensity
    arm_separation: float = 10e-2# in m
    particles: pd.DataFrame = None

    def measure_diameters(self):
        detected_particles = self.amplitude.intensity.measure_xy_diameters()

        self.xy_diameters = detected_particles
        
        return detected_particles
    
    def plot(self, detector=None, cloud=None, plot_outlines=False, **kwargs):
        
        plot = self.amplitude.intensity.plot(**kwargs)

        if plot_outlines:
            ax = plt.gca()
            plot_outline(self.get_focused_image(cloud, detector).amplitude.intensity.T<0.1, ax)

        return plot
    
    def get_focused_image(self, cloud, detector):
        if cloud is None:
            raise ValueError("Cloud must be specified")

        if detector is None:
            detector = Detector(self.detector_position, self.arm_separation)
        else:
            detector.position = self.detector_position

        primary_index = self.particles[self.particles.primary].index[0]
        cloud.particles["primary"] = cloud.particles.index == primary_index
        objects, _ = cloud.take_image(detector, distance=self.amplitude.shape[1] * self.amplitude.pixel_size, use_focus=True, separate_particles=True, primary_only=True)
        del cloud.particles["primary"]

        return objects[0]
        

@dataclass
class Particle:
    diameter: float # in m
    position: tuple[float, float, float] # (x, y, z) in m