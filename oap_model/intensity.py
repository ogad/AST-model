# Angular Spectrum Theory model
# Authour: Oliver Driver
# Date: 22/05/2023

# Standard library imports
from dataclasses import dataclass
from typing import Generator, Self

# Package imports
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm, TwoSlopeNorm
from scipy import ndimage

def plot_outline(mapimg, ax=None):
    """From https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph/24540564#24540564"""
    if not mapimg.any():
        return
    
    mapimg = np.flip(mapimg, axis=0)

    # a vertical line segment is needed, when the pixels next to each other horizontally
    #   belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates 
    ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])

    # the same is repeated for horizontal segments
    hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan,np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))

    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)

    # now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    segments[:,0] = x0 + (x1-x0) * segments[:,0] / mapimg.shape[1]
    segments[:,1] = y0 + (y1-y0) * segments[:,1] / mapimg.shape[0]

    # and now there isn't anything else to do than plot it
    ax.plot(segments[:,0], segments[:,1], color=(1,0,0,.5), linewidth=1)

@dataclass
class AmplitudeField:
    """A class to represent an amplitude field.

    Uses the numpy array as a base class, adding the pixel size attribute, storing
    a grid of phases from a single interaction

    Args:
        np.ndarray: The amplitude field
    """

    field: np.ndarray
    pixel_size: float = 10e-6

    @property
    def phase(self):
        """The phase field."""
        return np.fft.fft2(self.field)

    @property
    def intensity(self):
        """The intensity field."""
        return IntensityField(np.abs(self.field)**2, pixel_size=self.pixel_size)

    def embed(self, single_particle_amplitude, particle, detector_position): 
        """Embed the intensity profile of a particle into the total intensity array."""

        # TODO: this should take embed_iloc, rather than detector and particle positions...

        # vector from particle to detector
        pcle_from_detector = particle.position - detector_position

        
        # index of particle centre in total_intensity
        # detector is at x = self.shape[0]/2, y = 0
        x_index = int(pcle_from_detector[0] / self.pixel_size + self.field.shape[0]/2)
        y_index = self.field.shape[1] - int(pcle_from_detector[1] / self.pixel_size)
        embed_extent = [
            x_index - int(single_particle_amplitude.field.shape[0]/2), 
            x_index - int(single_particle_amplitude.field.shape[0]/2) + single_particle_amplitude.field.shape[0], 
            y_index - int(single_particle_amplitude.field.shape[1]/2), 
            y_index - int(single_particle_amplitude.field.shape[1]/2) + single_particle_amplitude.field.shape[1]
        ]

        amplitude_shape = single_particle_amplitude.field.shape

        # Check pixel sizes are consistent
        if single_particle_amplitude.pixel_size != self.pixel_size:
            raise ValueError(f"Pixel sizes of single_particle_amplitude and self must be the same.\nSingle particle: {single_particle_amplitude.pixel_size} m, Total: {self.pixel_size} m")

        # determine the bounds of the total intensity array to embed the particle intensity in
        # "do it to the edge, but not over the edge"
        embed_iloc = [x_index, y_index]
        total_min, total_max, single_min, single_max = [None, None], [None, None], [None, None], [None, None]
        for axis in [0,1]:
            total_min[axis] = 0 if embed_extent[axis*2] < 0 else embed_extent[axis*2]
            single_min[axis] = int(amplitude_shape[axis]/2) - embed_iloc[axis] if embed_extent[axis*2] < 0 else 0
            total_max[axis] = self.field.shape[axis] if embed_extent[axis*2+1] > self.field.shape[axis] else embed_extent[axis*2+1]
            single_max[axis] = amplitude_shape[axis] - (embed_extent[axis*2+1] - self.field.shape[axis]) if embed_extent[axis*2+1] > self.field.shape[axis] else amplitude_shape[axis]


        # check for the non-overlapping case
        for axis in [0,1]:
            if total_min[axis] > self.field.shape[axis] or total_max[axis] < 0:
                return self

        new_amplitude = single_particle_amplitude.field[single_min[0]:single_max[0], single_min[1]:single_max[1]]
        self.field = self.field
        self.field[total_min[0]:total_max[0], total_min[1]:total_max[1]] *= new_amplitude

@dataclass
class IntensityField:
    """A class to represent an intensity field.

    Uses the numpy array as a base class, adding the pixel size attribute, plotting
    methods, and measurement methods.

    Args:
        np.ndarray: The intensity field
    """

    field: np.ndarray
    pixel_size: float = 10e-6
    
    def plot(self, ax:plt.Axes=None, axis_length:float=None, grayscale_bounds:list[float]=None, colorbar:bool=False, **kwargs) -> plt.Axes:
        """Plot the intensity field.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            axis_length(float): The axis_length of the object in micrometres.
            grayscale_bounds (list): The list of bounds for grayscale bands. When None, the intensity is plotted.

        Returns:
            matplotlib.image.Axes: The plotted image.
        """
        if axis_length is not None:
            axis_length_px = axis_length * 1e-6 // self.pixel_size
            if axis_length_px > min(self.field.shape):
                to_plot = np.pad(self.field, int(axis_length_px - min(self.field.shape)), "constant", constant_values=(1,1))
            else:
                to_plot = self.field
            
            to_plot = to_plot[
                int((to_plot.shape[0] - axis_length_px) //
                    2):int((to_plot.shape[0] + axis_length_px) // 2),
                int((to_plot.shape[1] - axis_length_px) //
                    2):int((to_plot.shape[1] + axis_length_px) // 2),
            ]
        else:
            to_plot = self.field

        if ax is None:
            y_to_x_ratio = to_plot.shape[1] / to_plot.shape[0]
            fig, ax = plt.subplots(figsize = (5, 5 * y_to_x_ratio))

        if grayscale_bounds is not None:
            # Replace pixel values to the next-highest grayscale bound
            bounded = np.zeros_like(to_plot)
            # to_plot_values = []
            # for i, bound in enumerate(sorted(grayscale_bounds)):
            #     current_bound = (to_plot < bound) & (bounded == 0)
            #     to_plot = np.where(current_bound, i, to_plot)
            #     bounded = np.where(current_bound, 1, bounded)
            #     to_plot_values.append(i)
            
            # to_plot = np.where((bounded == 0), len(grayscale_bounds), to_plot)
            cmap = plt.cm.viridis
            cmaplist = [cmap(val) for val in range(cmap.N)]
            cmap = plt.cm.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
            
            bounds = sorted(grayscale_bounds)
            bounds = [0] + bounds + [2]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            kwargs["cmap"] = cmap
            kwargs["norm"] = norm
        else:
            kwargs["norm"] = plt.cm.colors.Normalize(vmin=0, vmax=2)


        xlen, ylen = np.array(to_plot.shape) * self.pixel_size * 1e6
        ax_image = ax.imshow(to_plot.T, extent=[0,xlen, 0, ylen], **kwargs)

        if colorbar:
            cax = ax.inset_axes([1.02, 0, 0.05, 1])
            cb = plt.colorbar(ax_image, ax=ax, cax=cax)
            if grayscale_bounds:
                cax.set_yticks(grayscale_bounds, [f"{bound}$I_0$" for bound in grayscale_bounds])
            else:
                cax.set_yticks([0, 1, 2], ["$0$", "$I_0$", "$2I_0$"])
            cax.set_facecolor("white")
            cax.set_alpha(0.5)
        ax.set_xlabel("$x$ (µm) (along detector)")
        ax.set_ylabel(r"$y-y_{\mathrm{det}}$ (µm)")
        ax.set_aspect("equal")

        return ax_image

    @staticmethod
    def _measure_xy_diameter(labelled_image, label) -> float:
        """Measure the diameter of a labelled region in the image.
        
        Returns:
            float: The diameter in units of pixels."""

        # isolate only the region of interest
        region = labelled_image == label

        # find the maximum extent in the x and y directions
        x_extent = np.sum(region, axis=0).max()
        y_extent = np.sum(region, axis=1).max()

        # return the average of the two extents in micrometres
        return (x_extent + y_extent) / 2
    
    @staticmethod
    def _measure_circle_equivalent_diameter(labelled_image, label) -> float:
        """Measure the diameter of a labelled region in the image.
        
        Returns:
            float: The diameter in units of pixels."""

        # isolate only the region of interest
        region = labelled_image == label

        n_pixels = np.sum(region)
        ce_diameter = 2 * np.sqrt(n_pixels / np.pi)

        # circle equivalent diameter in number of pixels
        return ce_diameter
    
    def _measure_position(self, labelled_image, label) -> tuple:
        region = labelled_image == label

        # calculate the position of the largest region
        position = [np.round(coords.mean() * self.pixel_size * 1e6) for coords in  np.where(region)]

        return tuple(position)

    def measure_diameters(self, threshold=0.5, bounded=False, filled=False, diameter_method="xy") -> dict:
        """Measure the diameters of all connected regions in the image.
        
        Returns:
            dict: The list of (diameter/µm, position/px) couples.
        """
        # threshold the image at 50% of the initial intensity
        thresholded_image = self.field < threshold

        diameter_method = getattr(self, f"_measure_{diameter_method}_diameter")

        if filled and not bounded:
            raise Exception("Cannot use filled without bounded")
        if bounded:
            labeled_image = np.zeros_like(thresholded_image)
            threshold_pixels = np.where(thresholded_image)
            if filled:
                labeled_image[
                    threshold_pixels[0].min(): threshold_pixels[0].max()+1,
                    threshold_pixels[1].min(): threshold_pixels[1].max()+1,
                ] = 1
            else:
                labeled_image[threshold_pixels] = 1
            n_labels = 1
        else:
            # iterate over connected regions
            labeled_image, n_labels = ndimage.label(thresholded_image, structure=np.ones((3, 3)))

        diameters = {}
        for label in range(1, n_labels + 1):
            diameter = diameter_method(labeled_image, label) * self.pixel_size * 1e6
            position = self._measure_position(labeled_image, label)
            
            diameters[position] = diameter

        # return dictionary of diameters in micrometres
        return diameters
    
    def frames(self, threshold=0.5) -> Generator[Self, None, None]:
        """Split the intensity field into frames.

        Args:
            threshold (float, optional): The threshold to use to split the frames. Defaults to 0.5.

        Yields:
            Generator[Self, None, None]: The intensity field for each frame.
        """
        frame_rows = np.where((self.field < threshold).any(axis=0))[0]
        if frame_rows.size == 0:
            return
        frames_indices = np.split(frame_rows, np.where(np.diff(frame_rows) != 1)[0] + 1)
        for frame in frames_indices:
            frame_field = self.field[:, frame[0]:frame[-1]+1]
            frame_intensityfield = IntensityField(frame_field, self.pixel_size)
            yield frame[0], frame_intensityfield
        # if not frames_indices:
        #     yield 0, self

    def n_pixels_depletion_range(self, min_dep:float, max_dep:float) -> int:
        """Calculate the number of pixels in the depletion range

        Args:
            min_dep (float): The minimum depletion value.
            max_dep (float): The maximum depletion value.

        Returns:
            int: The number of pixels in the depletion range.
        """
        min_intensity_counted = 1 - max_dep
        max_intensity_counted = 1 - min_dep

        n_pixels = np.sum((self.field <= max_intensity_counted)
                          & (self.field > min_intensity_counted))

        return n_pixels


