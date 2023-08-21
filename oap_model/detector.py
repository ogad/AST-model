# Detector model
# Author: Oliver Driver
# Date: 29/06/2023

from dataclasses import dataclass
from enum import Enum
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .intensity import plot_outline, AmplitudeField, IntensityField
from .diameters import measure_diameters

@dataclass
class Detector:
    position: tuple[float, float, float] # (x, y, z) in m
    arm_separation: float = 10e-2 # in m
    n_pixels: int = 128
    pixel_size: float =10e-6# in m
    wavelength: float = 658e-9 
    detection_length: float = np.inf #in m; the region in the middle of the detector that is used for detection. If np.inf, the whole detector is used.


class ImageFilter(Enum):
    PRESENT_HALF_INTENSITY = 1
    NO_EDGE_HALF_INTENSITY = 2
    PRIMARY_Z_CONFINED = 3

    def __call__(self, image: 'IntensityField'):
        if self == ImageFilter.PRESENT_HALF_INTENSITY:
            return image.field.min() <= 0.5
        elif self == ImageFilter.NO_EDGE_HALF_INTENSITY:
            return np.concatenate([image.field[0,:], image.field[-1,:]]).min() > 0.5
        # elif self == ImageFilter.PRIMARY_Z_CONFINED:
        #     return abs(image.particles.position[image.particles.primary].iloc[0][2] - image.detector_position[2]) < (image.amplitude.pixel_size * image.amplitude.field.shape[0] / 2)
        else:
            raise NotImplementedError(f"Image filter {self} not implemented")

@dataclass
class DiameterSpec: # TODO: implement custom threshold value.
    diameter_method: str = "circle_equivalent"
    edge_filter: bool = True
    framed: bool = True
    min_sep: float = None # in m #TODO: check units; enable time units
    bound: bool = True
    filled: bool = False
    z_confinement: bool = False
    c:float = 8.

    def __post_init__(self):
        if self.filled and not self.bound:
            raise ValueError("Can only be filled if bound.")
        if not self.framed and self.min_sep is not None:
            raise ValueError("min_sep can only be specified if framed")
        
    @property
    def filters(self):
        filters = [ImageFilter.PRESENT_HALF_INTENSITY]
        if self.edge_filter:
            filters.append(ImageFilter.NO_EDGE_HALF_INTENSITY)
        # if self.z_confinement:
        #     filters.append(ImageFilter.PRIMARY_Z_CONFINED)
        return filters


@dataclass
class ImagedRegion:
    """A region illuminated by the laser beam at a single instant."""
    # orthogonal_vector: np.ndarray = (0,0,1) # in m
    detector_position: np.ndarray # in m
    amplitude: AmplitudeField # relative intensity
    arm_separation: float = 10e-2# in m
    particles: pd.DataFrame = None

    @property
    def xlims(self):
        array_length = self.amplitude.pixel_size * self.amplitude.field.shape[0]
        return self.detector_position[0] + np.array([-array_length/2, array_length/2])
    
    def trim_blank_space(self):
        """Trim the blank space from the start and end of the image."""
        # trim start
        for i in range(self.amplitude.field.shape[1]):
            if (self.amplitude.field[:,i] < 0.9).any():
                self.amplitude.field = self.amplitude.field[:,i:]
                self.amplitude.intensity.field = self.amplitude.intensity.field[:,i:]
                break
        # trim end
        for i in range(self.amplitude.field.shape[1]-1, -1, -1):
            if (self.amplitude.field[:,i] < 0.9).any():
                self.amplitude.field = self.amplitude.field[:,:i+1]
                self.amplitude.intensity.field = self.amplitude.intensity.field[:,:i+1]
                break


    def get_frames_to_measure(self, spec, **kwargs) -> list[tuple[tuple[float, float], IntensityField]]:
        if spec.framed:
            # split image into "frames" separated by empty rows.
            frames = list(self.amplitude.intensity.frames())
        else:
            frames = [(0, self.amplitude.intensity)]

        if frames == []:
            return frames

        # Remove frames that aren't separated by at least min_sep (distance in m) #TODO: check units
        if spec.min_sep is not None:
            y_extents = [(self.y_values[istart], self.y_values[istart+frame.field.shape[1]-1]) for istart, frame in frames]
            # sort frames and y_extents on y_extent[0]
            zip_locs_frames = list(zip(y_extents, frames))
            zip_locs_frames.sort(key=lambda x: x[0][0])
            to_remove = []
            for i, (y_extent, frame) in enumerate(zip_locs_frames):
                if i == 0:
                    continue
                if y_extent[0] - zip_locs_frames[i-1][0][1] < spec.min_sep:
                    # mark for removal
                    to_remove.append(i)
                    to_remove.append(i-1)
            # remove duplicates
            to_remove = list(set(to_remove))
            for i in sorted(to_remove, reverse=True):
                del zip_locs_frames[i]
            # unzip
            frames = [frame for _, frame in zip_locs_frames]

        # replace index with y extremes
        frames = [((self.y_values[istart], self.y_values[istart+frame.field.shape[1]]), frame) for istart, frame in frames]
        return frames
    
    def measure_diameters(self, spec: DiameterSpec = DiameterSpec(), **kwargs):
        logging.warn("ImagedRegion.measure_diameters only considers one image at a time; use DetectorRun.measure_diameters to consider inter-image separation.")
        detected_particles = measure_diameters(self, spec, **kwargs)
        return detected_particles
    
    def plot(self, detector=None, cloud=None, plot_outlines=False,**kwargs):
        
        plot = self.amplitude.intensity.plot(**kwargs)

        if plot_outlines:
            ax=kwargs.get("ax")
            if ax is None:
                ax = plt.gca()
            for image in self.get_focused_image(cloud, detector):
                plot_outline(image.amplitude.intensity.field.T<0.1, ax)

        return plot
    
    def get_focused_image(self, cloud, detector):
        if cloud is None:
            raise ValueError("Cloud must be specified")

        if detector is None:
            detector = Detector(self.detector_position, self.arm_separation)
        else:
            detector.position = self.detector_position

        if "primary" in self.particles.columns:
            primary_index = self.particles[self.particles.primary].index[0]
            cloud.particles["primary"] = cloud.particles.index == primary_index
            focused_run = cloud.take_image(detector, distance=self.amplitude.field.shape[1] * self.amplitude.pixel_size, use_focus=True, primary_only=True)
            cloud.particles.drop("primary", axis=1, inplace=True)
            yield focused_run.images[0]
        else:
            for particle in self.particles.index:
                cloud.particles.loc[particle, "primary"] = True
            focused_image = cloud.take_image(detector, distance=self.amplitude.field.shape[1] * self.amplitude.pixel_size, use_focus=True, primary_only=True)
            cloud.particles.drop("primary", axis=1, inplace=True)
            yield focused_image

    @property
    def distance(self):
        return self.amplitude.field.shape[1] * self.amplitude.pixel_size
    
    @property
    def start(self):
        """The start of the region in the detector's reference frame, note that this is the high y value."""
        return self.detector_position[1] 
    
    @property
    def end(self):
        """The end of the region in the detector's reference frame, note that this is the low y value."""
        return self.detector_position[1]- self.amplitude.pixel_size * self.amplitude.field.shape[1] 
    
    @property
    def y_values(self): # y values decrease
        """The y values of the detector pixels, aligned so the y value at index i is the y value of the pixel at index i."""
        n_pixels = self.amplitude.field.shape[1]
        range = np.arange(self.start, self.end - self.amplitude.pixel_size/2, -1*self.amplitude.pixel_size)
        return range[:n_pixels+1]

