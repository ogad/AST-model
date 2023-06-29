# Detector model
# Author: Oliver Driver
# Date: 29/06/2023

from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ast_model import plot_outline, AmplitudeField

@dataclass
class Detector:
    position: tuple[float, float, float] # (x, y, z) in m
    arm_separation: float = 10e-2 # in m
    n_pixels: int = 128
    pixel_size: float =10e-6# in m


@dataclass
class ImagedRegion:
    """A region illuminated by the laser beam at a single instant."""
    # orthogonal_vector: np.ndarray = (0,0,1) # in m
    detector_position: np.ndarray # in m
    amplitude: AmplitudeField # relative intensity
    arm_separation: float = 10e-2# in m
    particles: pd.DataFrame = None

    def measure_diameters(self, type="xy"):
        if type == "xy":
            # XY diameter at 0.5I_0 intensity threshold
            # Default behviour
            detected_particles = self.amplitude.intensity.measure_xy_diameters()
            self.xy_diameters = detected_particles
        else:
            raise NotImplementedError("Only xy diameters are currently supported")
        
        return detected_particles
    
    def plot(self, detector=None, cloud=None, plot_outlines=False, **kwargs):
        
        plot = self.amplitude.intensity.plot(**kwargs)

        if plot_outlines:
            ax = plt.gca()
            plot_outline(self.get_focused_image(cloud, detector).amplitude.intensity.field.T<0.1, ax)

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
        focused_run = cloud.take_image(detector, distance=self.amplitude.field.shape[1] * self.amplitude.pixel_size, use_focus=True, separate_particles=True, primary_only=True)
        del cloud.particles["primary"]

        return focused_run.images[0]

@dataclass
class DetectorRun:
    detector: Detector
    images: list[ImagedRegion]
    # particles: pd.DataFrame # The particles illuminated by the laser beam (not necessarily detected) 
    distance: float # in m


    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            run =  pickle.load(f)

        return run

    def measure_diameters(self, type="xy"):
        diameters = []
        for image in self.images:
            diameter_dict = image.measure_diameters(type=type)
            diameters = diameters + list(diameter_dict.values())
        
        return diameters