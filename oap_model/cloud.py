# Cloud model
# Author: Oliver Driver
# Date: 13/06/2023

from copy import deepcopy
from dataclasses import dataclass
import random
import pandas as pd
import logging
import numpy as np
from tqdm.autonotebook import tqdm

import matplotlib.pyplot as plt

from .psd import CrystalModel, GammaPSD
from .intensity import AmplitudeField
from .detector import Detector, ImagedRegion
from .detector_run import DetectorRun


def process_image_no(image_no, particles_to_model, detector_position, detector, cloud, use_focus, binary_output):
    particles_in_image = particles_to_model[particles_to_model["image_no"]==image_no]

    # offset detector s.t. the end is 5*last_particle.diameter after the "last" particle, and the start is 5*first_particle.diameter before the first particle
    last_y = particles_in_image.iloc[0].y_value # low
    last_diameter = particles_in_image.iloc[0].diameter
    first_y = particles_in_image.iloc[-1].y_value # high
    first_diameter = particles_in_image.iloc[-1].diameter
    new_detector_y = last_y - 5*last_diameter
    new_distance = (first_y + 5*first_diameter) - (last_y - 5*last_diameter)
    y_offset = new_detector_y - detector_position[1]
    z_offset = particles_in_image.iloc[0].position[2] - (detector_position[2] + detector.arm_separation/2) if use_focus else 0 #TODO: check this after refactoring to own function
    image = cloud.process_imaged_region(particles_to_model[particles_to_model["image_no"]==image_no], detector, new_distance, offset=np.array([0, y_offset, z_offset]), use_focus=False, binary_output=binary_output)
    if (image is not None) and (image.amplitude.intensity.field <= 0.5).any():
        # image.particles["primary"] = image.particles.index == particles_in_image.index[0] #TODO: reimpliment in a non set on slice way
        return image


@dataclass
class CloudVolume:
    psd: GammaPSD
    dimensions: tuple[float, float, float] # (x, y, z) in m
    random_seed: int = 42

    def _generate_positions(self, resolution, n_particles, y_offset=0):
        """Generate a random position within the cloud volume."""
        if self.dimensions[0] > 1e3 or self.dimensions[2] > 1e3:
            raise Exception("X and Z dimensions cannot be extended beyond 1km.")
        elif self.dimensions[1] > 1e3:
            possible_offsets = np.arange(0, self.dimensions[1], 1e3)
            weights = np.diff(possible_offsets)
            weights = np.append(weights, self.dimensions[1])
            y_offset = np.random.choice(possible_offsets, p=weights/np.sum(weights), size=(n_particles))
        else:
            y_offset = np.zeros((n_particles))

        rng = np.random.default_rng()
        xs = rng.integers(0, int(self.dimensions[0]//resolution), size=(n_particles))
        ys = rng.integers(0, int(self.dimensions[1]//resolution), size=(n_particles)) + y_offset
        zs = rng.integers(0, int(self.dimensions[2]//resolution), size=(n_particles))
        return np.array([xs, ys, zs]).T * resolution
    
    def _generate_angles(self, n_particles):
        """Generate a random angle for each particle."""
        rng = np.random.default_rng()
        thetas = rng.uniform(0, 2*np.pi, size=(n_particles))
        # generate phis weighted by sin(theta)
        sin_phis = rng.uniform(0, 1, size=(n_particles))
        phis = np.arcsin(sin_phis)

        return np.array([thetas, phis]).T
    
    def plot_from_run(self, run: DetectorRun, near_coord=None, near_length=2e-3, ylims=None, ax=None, **kwargs):

        new_detector_y = np.min(ylims) if ylims is not None else near_coord[1] - near_length/2
        distance = abs(ylims[0] - ylims[1]) if ylims is not None else near_length

        # take the image
        detector = deepcopy(run.detector)
        detector.position[1] = new_detector_y
        image = self.take_image(detector, distance=distance, single_image=True)

        # plot the image
        if ax is None:
            ax = plt.gca()
        image.images[0].plot(ax=ax, **kwargs)

    def __post_init__(self):
        logging.info("Initialising cloud volume")

        self.particles = None
        self.generate_particles()

    def generate_particles(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if self.particles is not None:
            raise ValueError("Particles already generated.")

        logging.info(f"Generating grid of dimensions: {[int(dim / 1e-6) for dim in self.dimensions]} points.")
        # # raise warning if any dimension will have more than 2e9 points
        # if any([dim > 1e3 for dim in self.dimensions]):
        #     logging.warn("One or more dimensions is too great to grid at 2µm resolution.")

        # Generate the particles
        # self.particles = pd.DataFrame(columns=["diameter", "position", "model"])
        logging.info(f"Generating {self.n_particles} particles")

        positions = self._generate_positions(1e-6, self.n_particles)
        diameters, models = self.psd.generate_diameters(self.n_particles)

        angles = self._generate_angles(self.n_particles)

        self.particles = pd.DataFrame({"diameter": diameters, "position":list(map(tuple, positions)), "angle":list(map(tuple, angles)), "model": models})
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
    
    def particles_in_illuminated_region(self, detector: Detector, distance:float=10e-6, offset: np.ndarray = np.array([0,0,0]) ):
        detector_position = detector.position + offset

        
        beam_width = 8e-3 - (128 * 10e-6) + (detector.n_pixels * detector.pixel_size) # leave the same padding
        beam_length = 2e-3 - (10e-6) + (detector.pixel_size) # leave the same padding

        # check which particles are somewhat within within the illuminated region
        # Illuminated region is defined as the 8mm x 2mm x arm_separation region aligned with the orthogonal vector pointing towards the detector
        # TODO: These may also need to also detect particles whose centres are outside the illuminated region, but whos edges are inside it.
        particle_from_detector = np.stack(self.particles.position.values) - detector_position

        is_in_illuminated_region_x = abs(particle_from_detector[:,0]) < (beam_width/2)
        is_in_illuminated_region_y = (particle_from_detector[:,1] < (beam_length/2 + distance)) & (particle_from_detector[:,1] > -1*beam_length/2)
        is_in_illuminated_region_z = (particle_from_detector[:,2]< detector.arm_separation) & (particle_from_detector[:,2] > 0 )

        in_illuminated_region = is_in_illuminated_region_x & is_in_illuminated_region_y & is_in_illuminated_region_z
        if not in_illuminated_region.any():
            # No particles are in the illuminated region.
            return None
        
        stereo_observed = np.abs(particle_from_detector[:, 2] - detector.arm_separation/2) < detector.detection_length/2
        
        illuminated_particles = self.particles[in_illuminated_region].copy()
        illuminated_particles["stereo_observed"] = stereo_observed[in_illuminated_region]

        to_test_stereo = illuminated_particles[illuminated_particles.stereo_observed]
        for particle in to_test_stereo.itertuples():# TODO: this could be improved by accounting for e.g. postition in detector
            model = particle.model.get_generator()(particle, wavenumber=2*np.pi/detector.wavelength, pixel_size=detector.pixel_size)
            if (model.process(particle_from_detector[particle.Index, 0]).intensity.field > 0.5).all():
                illuminated_particles.loc[particle.Index, "stereo_observed"] = False
        

        return illuminated_particles
        
    
    def take_image(self, detector: Detector, distance:float=10e-6, offset: np.ndarray = np.array([0,0,0]), single_image: bool=False, use_focus: bool=False, primary_only: bool=False, detection_condition: callable=None, binary_output:bool = False) -> DetectorRun | ImagedRegion:
        """Take an image using repeated detections along the y-axis.

        Detector is aligned with x-axis, and the y-axis is the direction of travel.
        The z-axis is the focal axis.
        The detector position is the position of the detector centre during the final detection.
        The model_generator is a callable taking the diameter of the particle in microns and returning an ASTModel.
        
        """
        detector_position = detector.position + offset

        particles_to_model = self.particles_in_illuminated_region(detector, distance, offset).copy()
        if primary_only:
            particles_to_model = particles_to_model[particles_to_model.primary]

        # for dense or large image regions, do some filtering to reduce the number of particles to model
        if len(particles_to_model) > 10: 
            models = particles_to_model.model.unique()
            min_diameter = {model: model.min_diameter(pixel_size=detector.pixel_size) for model in models}
            particles_to_model = particles_to_model[particles_to_model.apply(lambda particle: particle.diameter >= min_diameter[particle.model], axis=1)]

        # define images to take based on particles' separation in y
        particles_to_model["y_value"] = particles_to_model.position.apply(lambda pos: pos[1])
        particles_to_model = particles_to_model.sort_values(by="y_value", inplace=False)
        particles_to_model["y_separation"] = particles_to_model["y_value"].diff()

        if single_image:
            return DetectorRun(detector, [self.process_imaged_region(particles_to_model, detector, distance, offset, use_focus=use_focus)], distance)
        
        image_nos = []
        for particle in particles_to_model.itertuples():
            if np.isnan(particle.y_separation):
                image_nos.append(0)
                prev_diameter = particle.diameter
                continue
            
            if particle.y_separation  < min(10*prev_diameter, 10*particle.diameter):
                image_nos.append(image_nos[-1])
            else:
                image_nos.append(image_nos[-1] + 1)

            prev_diameter = particle.diameter

        particles_to_model["image_no"] = image_nos

        args = [(image_no, particles_to_model, detector_position, detector, self, use_focus, binary_output) for image_no in image_nos]

        # with multiprocessing.Pool(1) as pool:
        #     images = pool.starmap_async(process_image_no, args)
        #     pool.close()
        #     pool.join()
        #     images = [image for image in images.get() if image is not None]

        # loop = asyncio.get_event_loop()
        # looper = asyncio.gather(*[process_image_no(image_no) for image_no in range(image_nos[-1]+1)])
        # images = loop.run_until_complete(looper)
        # images = [image for image in images if image is not None]


        images = [] 
        for image_no in tqdm(range(image_nos[-1]+1), leave=False, smoothing=0.1):
            particles_in_image = particles_to_model[particles_to_model["image_no"]==image_no]

            # offset detector s.t. the end is 5*last_particle.diameter after the "last" particle, and the start is 5*first_particle.diameter before the first particle
            last_y = particles_in_image.iloc[0].y_value # low
            last_diameter = particles_in_image.iloc[0].diameter
            first_y = particles_in_image.iloc[-1].y_value # high
            first_diameter = particles_in_image.iloc[-1].diameter
            new_detector_y = last_y - 5*last_diameter
            new_distance = (first_y + 5*first_diameter) - (last_y - 5*last_diameter)
            y_offset = new_detector_y - detector_position[1]
            z_offset = particles_in_image.iloc[0].position[2] - (detector_position[2] + detector.arm_separation/2) if use_focus else 0 #TODO: check this after refactoring to own function
            image = self.process_imaged_region(particles_to_model[particles_to_model["image_no"]==image_no], detector, new_distance, offset=np.array([0, y_offset, z_offset]), use_focus=False, binary_output=binary_output)
            if (image is not None) and (image.amplitude.intensity.field <= 0.5).any():
                # image.particles["primary"] = image.particles.index == particles_in_image.index[0] #TODO: reimpliment in a non set on slice way
                images.append(image)
        
        return DetectorRun(detector, images, distance)

        # if separate_particles: #TODO: This should be default behaviour
        #     # take an image of each particle individually
        #     images = []
        #     length_iterations = len(particles_to_model)
        #     for particle in tqdm(particles_to_model.itertuples(), total=length_iterations, leave=False):
        #         y_offset = particle.position[1] - detector_position[1] - 5 * particle.diameter
        #         z_offset = particle.position[2] - (detector_position[2] + detector.arm_separation/2) if use_focus else 0
        #         particle_run = self.take_image(detector, 10*particle.diameter, offset=np.array([0, y_offset, z_offset]), use_focus=use_focus)
        #         if particle_run is not None:
        #             particle_run.images[0].particles["primary"] = particle_run.images[0].particles.index == particle.Index

        #             images += particle_run.images

        #     run = DetectorRun(detector, images, distance)
        #     return run 
    
    def process_imaged_region(self, particles, detector, distance, offset, use_focus=False, binary_output=False): #TODO: can be static, probably belongs on ImagedRegion.
        detector_position = detector.position + offset
        n_rows = int(distance / detector.pixel_size)

        # get the intensity profile at the given z value for each illuminated particle
        total_amplitude = AmplitudeField(np.ones((detector.n_pixels, n_rows), dtype=np.complex128), pixel_size=detector.pixel_size)
        
        # particles["x_index"] = particles.apply(
        #     lambda particle: int((particle["position"] - detector_position)[0] / total_amplitude.pixel_size) + int(total_amplitude.field.shape[0]/2),
        #     axis=1)
        # particles["y_index"] = particles.apply(
        #     lambda particle: int((particle["position"] - detector_position)[1] / total_amplitude.pixel_size),
        #     axis=1)

        
        generators = {model: model.get_generator() for model in particles.model.unique()}


        n_pcles = len(particles)
        for particle in tqdm(particles.itertuples(), total=n_pcles, leave=False, disable=n_pcles<10):
            ast_model = generators[particle.model](particle, wavenumber=2*np.pi/detector.wavelength, pixel_size=detector.pixel_size)
            if use_focus:
                amplitude_at_particle_xy = ast_model.process(0)
            else:
                amplitude_at_particle_xy = ast_model.process(particle.position[2] - detector_position[2] - detector.arm_separation/2)

            total_amplitude.embed(amplitude_at_particle_xy, particle, detector_position)
        
        if binary_output:
            total_amplitude.field = np.where(total_amplitude.intensity.field > 0.5, 1, 0).astype(np.int8)
        return ImagedRegion(detector_position, total_amplitude, particles=particles)

    def set_model(self, shape:CrystalModel):
        self.particles["model"] = shape
        self.psd.model = shape