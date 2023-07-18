# Cloud model
# Author: Oliver Driver
# Date: 13/06/2023

from copy import deepcopy
from dataclasses import dataclass
import random
from tkinter import Image
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm
from enum import Enum

import matplotlib.pyplot as plt

from psd_ast_model import GammaPSD
from ast_model import ASTModel, AmplitudeField
from detector_model import Detector, ImagedRegion
from detector_run import DetectorRun


@dataclass
class CloudVolume:
    psd: GammaPSD
    dimensions: tuple[float, float, float] # (x, y, z) in m

    def _generate_positions(self, resolution, n_particles):
        """Generate a random position within the cloud volume."""
        rng = np.random.default_rng()
        xs = rng.integers(0, int(self.dimensions[0]//resolution), size=(n_particles))
        ys = rng.integers(0, int(self.dimensions[1]//resolution), size=(n_particles))
        zs = rng.integers(0, int(self.dimensions[2]//resolution), size=(n_particles))
        return np.array([xs, ys, zs]).T * resolution
    
    def plot_from_run(self, run: DetectorRun, near_coord=None, near_length=2e-3, ylims=None, ax=None, **kwargs):

        new_detector_y = np.min(ylims) if ylims is not None else near_coord[1] - near_length/2
        distance = abs(ylims[0] - ylims[1]) if ylims is not None else near_length

        # take the image
        detector = deepcopy(run.detector)
        detector.position[1] = new_detector_y
        image = self.take_image(detector, distance=distance)

        # plot the image
        if ax is None:
            ax = plt.gca()
        image.plot(ax=ax, **kwargs)

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
        # self.particles = pd.DataFrame(columns=["diameter", "position", "model"])
        logging.info(f"Generating {self.n_particles} particles")

        positions = self._generate_positions(2e-6, self.n_particles)
        diameters, models = self.psd.generate_diameters(self.n_particles)

        self.particles = pd.DataFrame({"diameter": diameters, "position":list(map(tuple, positions)), "model": models})
        # for i in tqdm(range(self.n_particles), total=self.n_particles):
        #     particle = [diameters_models[i][0], positions[i, :], diameters_models[i][1]]
        #     self.particles.loc[i] = particle

    @property
    def n_particles(self):
        """Calculate the number of particles in the volume based on the PSD."""
        return int(self.volume * self.psd.total_number_density)

    @property
    def volume(self):
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    
    def take_image(self, detector: Detector, distance:float=10e-6, offset: np.ndarray = np.array([0,0,0])  ,separate_particles: bool=False, use_focus: bool=False, primary_only: bool=False, detection_condition: callable=None) -> DetectorRun | ImagedRegion:
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
        particle_from_detector = np.stack(self.particles.position.values) - detector_position

        is_in_illuminated_region_x = abs(particle_from_detector[:,0]) < (beam_width/2)
        is_in_illuminated_region_y = (particle_from_detector[:,1] < (beam_length/2 + n_images * detector.pixel_size)) & (particle_from_detector[:,1] > -1*beam_length/2)
        is_in_illuminated_region_z = (particle_from_detector[:,2]< detector.arm_separation) & (particle_from_detector[:,2] > 0 )

        in_illuminated_region = is_in_illuminated_region_x & is_in_illuminated_region_y & is_in_illuminated_region_z
        if not in_illuminated_region.any():
            # No particles are in the illuminated region.
            return None
        
        particles_to_model = self.particles[in_illuminated_region]
        if primary_only:
            particles_to_model = particles_to_model[particles_to_model.primary]

        if separate_particles: #TODO: This should be default behaviour
            # take an image of each particle individually
            images = []
            length_iterations = len(particles_to_model)
            for particle in tqdm(particles_to_model.itertuples(), total=length_iterations, leave=False):
                y_offset = particle.position[1] - detector_position[1] - 5 * particle.diameter
                z_offset = particle.position[2] - (detector_position[2] + detector.arm_separation/2) if use_focus else 0
                particle_image = self.take_image(detector, 10*particle.diameter, offset=np.array([0, y_offset, z_offset]), use_focus=use_focus)
                if particle_image is not None:
                    particle_image.particles["primary"] = particle_image.particles.index == particle.Index

                    images.append(particle_image)

            run = DetectorRun(detector, images, distance)
            return run 

        # get the intensity profile at the given z value for each illuminated particle
        total_amplitude = AmplitudeField(np.ones((detector.n_pixels, n_images), dtype=np.complex128), pixel_size=detector.pixel_size)
        
        particles = particles_to_model.copy()
        particles["x_index"] = particles.apply(
            lambda particle: int((particle["position"] - detector_position)[0] / total_amplitude.pixel_size) + int(total_amplitude.field.shape[0]/2),
            axis=1)
        particles["y_index"] = particles.apply(
            lambda particle: int((particle["position"] - detector_position)[1] / total_amplitude.pixel_size),
            axis=1)

        for particle in particles.itertuples():
            model_generator = particle.model.get_generator() if particle.model is not None else model_generator
            ast_model = model_generator(particle.diameter * 1e6, wavenumber=2*np.pi/detector.wavelength, pixel_size=detector.pixel_size)
            if use_focus:
                amplitude_at_particle_xy = ast_model.process(0)
            else:
                amplitude_at_particle_xy = ast_model.process(particle.position[2] - detector_position[2] - detector.arm_separation/2)

            total_amplitude.embed(amplitude_at_particle_xy, particle, detector_position)
        
        image = ImagedRegion(detector_position, total_amplitude, particles=particles)
        return image