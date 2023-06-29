# Cloud model
# Author: Oliver Driver
# Date: 13/06/2023

from dataclasses import dataclass
import random
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from enum import Enum

import matplotlib.pyplot as plt

from psd_ast_model import GammaPSD
from ast_model import ASTModel, AmplitudeField
from detector_model import Detector, ImagedRegion, DetectorRun

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
        self.random_state = random.getstate()

        self.particles = None
        self.generate_particles()

    def generate_particles(self):
        random.setstate(self.random_state)

        if self.particles is not None:
            raise ValueError("Particles already generated.")

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
    
    def take_image(self, detector: Detector, distance:float=10e-6, offset: np.ndarray = np.array([0,0,0])  ,separate_particles: bool=False, use_focus: bool=False, primary_only: bool=False, detection_condition: callable=None):
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
        # TODO: These may also need to also detect particles whose centres are outside the illuminated region, but whos edges are inside it.
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

        if separate_particles: #TODO: This should be default behaviour
            # take an image of each particle individually
            images = []
            length_iterations = len(self.particles[in_illuminated_region])
            for particle in tqdm(self.particles[in_illuminated_region].itertuples(), total=length_iterations, leave=False):
                if primary_only and not particle.primary:
                    continue
                y_offset = particle.position[1] - detector_position[1] - 5 * particle.diameter
                z_offset = particle.position[2] - (detector_position[2] + detector.arm_separation/2) if use_focus else 0
                particle_image = self.take_image(detector, 10*particle.diameter, offset=np.array([0, y_offset, z_offset]), use_focus=use_focus)
                if particle_image is not None:
                    particle_image.particles["primary"] = particle_image.particles.index == particle.Index

                    images.append(particle_image)

            run = DetectorRun(detector, images, distance)

            # particles = self.particles[in_illuminated_region].copy()
            # run = DetectorRun(detector, images, particles, distance)
            return run 
        


        # get the intensity profile at the given z value for each illuminated particle
        total_amplitude = AmplitudeField(np.ones((detector.n_pixels, n_images), dtype=np.complex128), pixel_size=detector.pixel_size)
        
        particles = self.particles[in_illuminated_region].copy()
        particles["x_index"] = particles.apply(
            lambda particle: int((particle["position"] - detector_position)[0] / total_amplitude.pixel_size) + int(total_amplitude.field.shape[0]/2),
            axis=1)
        particles["y_index"] = particles.apply(
            lambda particle: int((particle["position"] - detector_position)[1] / total_amplitude.pixel_size),
            axis=1)

        for particle in particles.itertuples():
            model_generator = particle.model.get_generator() if particle.model is not None else model_generator
            ast_model = model_generator(particle.diameter * 1e6, wavenumber=2*np.pi/detector.wavelength)
            amplitude_at_particle_xy = ast_model.process(particle.position[2] - detector_position[2] - detector.arm_separation/2)

            total_amplitude.embed(amplitude_at_particle_xy, particle, detector_position)
        
        image = ImagedRegion(detector_position, total_amplitude, particles=particles)
        return image