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
    wavelength: float = 658e-9 


@dataclass
class ImagedRegion:
    """A region illuminated by the laser beam at a single instant."""
    # orthogonal_vector: np.ndarray = (0,0,1) # in m
    detector_position: np.ndarray # in m
    amplitude: AmplitudeField # relative intensity
    arm_separation: float = 10e-2# in m
    particles: pd.DataFrame = None

    def measure_diameters(self, type="xy", **kwargs):
        if type == "xy":
            # XY diameter at 0.5I_0 intensity threshold
            # Default behviour
            detected_particles = self.amplitude.intensity.measure_xy_diameters(**kwargs)
            self.xy_diameters = detected_particles
        else:
            raise NotImplementedError("Only xy diameters are currently supported")
        
        return detected_particles
    
    def plot(self, detector=None, cloud=None, plot_outlines=False,**kwargs):
        
        plot = self.amplitude.intensity.plot(**kwargs)

        if plot_outlines:
            ax=kwargs.get("ax")
            if ax is None:
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

    def measure_diameters(self, type="xy", image_filter=lambda im: im.amplitude.intensity.field.min() <= 0.5):
        diameters = []
        for image in self.images:
            if not image_filter(image):
                continue
            diameter_dict = image.measure_diameters(type=type)
            diameters = diameters + list(diameter_dict.values())
        
        return diameters

    def plot(self, image_filter=lambda im: im.amplitude.intensity.field.min() < 0.5, **kwargs):
        images_to_plot = [image for image in self.images if image_filter(image)]
        
        n_plots = len(images_to_plot)
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5),sharex=True)
        for image, ax in zip(images_to_plot, axs.flatten()):
            image.plot(ax=ax, detector=None, **kwargs)


        n_bottom = n_plots % 3
        if n_bottom != 0:
            for ax in axs[-2][n_bottom:]:
                ax.xaxis.set_tick_params(labelbottom=True)
            for ax in axs[-1][n_bottom:]:
                ax.remove()
        return fig, axs
    
    def volume(self, diameter, c=8):
        sample_length = self.distance # m
        effective_array_width = self.detector.pixel_size * (self.detector.n_pixels - 1) - diameter # ? m: pixel_size * (n_pixels - 1) - diameter (parallel to array?)
        depth_of_field = min(self.detector.arm_separation, c * diameter**2 / (4 * self.detector.wavelength))# ? m ± cD^2/4λ; c = 8 ish for 2D-S. (from Gurganus Lawson 2018)
        sample_volume = sample_length * effective_array_width * depth_of_field # should strictly be integrated...
        return sample_volume
