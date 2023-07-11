# Detector run
# Author: Oliver Driver
# Date: 29/06/2023

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pickle

from ast_model import plot_outline, AmplitudeField, IntensityField
from detector_model import Detector, ImageFilter, DiameterSpec, ImagedRegion


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
    
    def get_frames_to_measure(self, spec, **kwargs) -> list[ImagedRegion]:
        """Returns a list of frames to measure, with the y extent of the frame and the frame itself."""
        frames = []
        image_filters = spec.filters
        for image in self.images:
            if not np.all([image_filter(image) for image_filter in image_filters]):
                continue
            frames = frames + list(image.get_frames_to_measure(spec, **kwargs))

        return frames

    def measure_diameters(self, spec=DiameterSpec(), **kwargs):
        frames = self.get_frames_to_measure(spec, **kwargs)

        frames.sort(key=lambda x: x[0][0])
        to_remove = []

        if spec.min_sep is not None:
            for i, ((ymin, ymax), frame) in enumerate(frames):
                    if i == 0:
                        continue
                    if ymin - frames[i-1][0][1] < spec.min_sep:
                        # mark for removal
                        to_remove.append(i)
                        to_remove.append(i-1)
            # remove duplicates
            to_remove = list(set(to_remove))
            for i in sorted(to_remove, reverse=True):
                del frames[i]

        kwargs["bounded"] = spec.bound
        kwargs["filled"] = spec.filled

        detected_particles = {}
        for (_,_), frame in frames:
            detected_particles = detected_particles | frame.measure_diameters(diameter_method=spec.diameter_method, **kwargs)

        diameters = list(detected_particles.values())
        return diameters

    def plot(self, image_filters: list[ImageFilter]=[ImageFilter.PRESENT_HALF_INTENSITY], **kwargs):
        images_to_plot = [image for image in self.images if np.all([image_filter(image) for image_filter in image_filters])]
        
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
        # TODO: in theory, parallel diameter can be different to DOF diameter (!)
        sample_length = self.distance # m
        effective_array_width = self.detector.pixel_size * (self.detector.n_pixels - 1) - diameter # ? m: pixel_size * (n_pixels - 1) - diameter (parallel to array?)
        depth_of_field = np.minimum(self.detector.arm_separation, c * diameter**2 / (4 * self.detector.wavelength))# ? m ± cD^2/4λ; c = 8 ish for 2D-S. (from Gurganus Lawson 2018)
        sample_volume = sample_length * effective_array_width * depth_of_field # should strictly be integrated...
        return sample_volume
    
    def overlaps(self):
        ends = [im.end for im in self.images]
        starts = [im.start for im in self.images]
        regions = list(zip(range(len(starts)), starts, ends))
        sorted_regions = sorted(regions, key=lambda x: x[1])

        overlaps = []
        for i in range(len(sorted_regions)-1):
            # if the end of the current region is after the start of the next region
            if sorted_regions[i][2] > sorted_regions[i+1][1]:
                # print("Overlap detected")
                overlaps.append((sorted_regions[i], sorted_regions[i+1]))
        
        for overlap in overlaps:
            intensity_1 = self.images[overlap[0][0]].amplitude.intensity.field
            intensity_2 = self.images[overlap[1][0]].amplitude.intensity.field
            
            y_vals_1 = self.images[overlap[0][0]].y_values
            y_vals_2 = self.images[overlap[1][0]].y_values

            # the end of 1 overlaps with the beginning of 2
            overlap_start = y_vals_1[0] #bigger value, at lesser index
            overlap_end = y_vals_2[-1] #smaller value, at greater index
            if overlap_start > overlap_end:
                logging.warning("Overlap start > overlap end.")
                continue
            
            # the index after the overlap in the first image
            overlap_end_index_1 = np.argwhere(y_vals_1 < overlap_end)[0][0]
            # the index after the overlap ends in the second image
            overlap_start_index_2 = np.argwhere(y_vals_2 > overlap_start)[-1][0]

            overlap_intensity_1 = intensity_1[:overlap_end_index_1]
            overlap_intensity_2 = intensity_2[overlap_start_index_2:]

            if (overlap_intensity_1 <0.9).any() or (overlap_intensity_2 < 0.9).any():
                logging.warning("Overlap intensity has some signal < 0.9.")
                continue
        return overlaps