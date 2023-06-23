# Angular Spectrum Theory model
# Authour: Oliver Driver
# Date: 22/05/2023

# Standard library imports
from dataclasses import dataclass
from copy import deepcopy

# Package imports
import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import ndimage
from skimage.transform import rescale

class AmplitudeField(np.ndarray):
    """A class to represent an amplitude field.

    Uses the numpy array as a base class, adding the pixel size attribute, storing
    a grid of phases from a single interaction

    Args:
        np.ndarray: The amplitude field
    """

    def __new__(cls, input_array, pixel_size=None):
        obj = np.asarray(input_array).view(cls)
        obj.pixel_size = pixel_size
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.pixel_size = getattr(obj, "pixel_size", None)

    @property
    def phase(self):
        """The phase field."""
        return np.fft.fft2(self)

    @property
    def intensity(self):
        """The intensity field."""
        return IntensityField(np.abs(self)**2, pixel_size=self.pixel_size)


class IntensityField(np.ndarray):
    """A class to represent an intensity field.

    Uses the numpy array as a base class, adding the pixel size attribute, plotting
    methods, and measurement methods.

    Args:
        np.ndarray: The intensity field
    """

    def __new__(cls, input_array, pixel_size=None):
        obj = np.asarray(input_array).view(cls)
        obj.pixel_size = pixel_size
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.pixel_size = getattr(obj, "pixel_size", None)

    def plot(self, ax:plt.Axes=None, axis_length:float=None, grayscale_bounds:list[float]=None, **kwargs) -> plt.Axes:
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
            to_plot = self[
                int((self.shape[0] - axis_length_px) //
                    2):int((self.shape[0] + axis_length_px) // 2),
                int((self.shape[1] - axis_length_px) //
                    2):int((self.shape[1] + axis_length_px) // 2),
            ]
        else:
            to_plot = self

        if ax is None:
            y_to_x_ratio = to_plot.shape[1] / to_plot.shape[0]
            fig, ax = plt.subplots(figsize = (5, 5 * y_to_x_ratio))

        if grayscale_bounds is not None:
            # Replace pixel values to the next-highest grayscale bound
            bounded = np.zeros_like(to_plot)
            for i, bound in enumerate(sorted(grayscale_bounds)):
                current_bound = (to_plot < bound) & (bounded == 0)
                to_plot = np.where(current_bound, i, to_plot)
                bounded = np.where(current_bound, 1, bounded)
            to_plot = np.where((bounded == 0), len(grayscale_bounds), to_plot)
        else:
            kwargs["norm"] = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)


        xlen, ylen = np.array(to_plot.shape) * self.pixel_size * 1e6
        ax_image = ax.imshow(to_plot.T, extent=[0,xlen, 0, ylen], **kwargs)
        plt.colorbar(ax_image, ax=ax)
        ax.set_xlabel("x/µm (Detector)")
        ax.set_ylabel("y/µm (Travelling in -y direction)")
        ax.set_aspect("equal")

        return ax_image

    def measure_xy_diameter(self) -> float:
        """Measure the diameter of the largest connected region in the image.
        
        Returns:
            float: The diameter in micrometres."""
        # threshold the image at 50% of the initial intensity
        thresholded_image = self < 0.5

        # isolate only the largest conncted region
        labeled_image, n_labels = ndimage.label(thresholded_image, structure=np.ones((3, 3)))
        label_counts = np.bincount(labeled_image.ravel())
        largest_label = np.argmax(label_counts[1:]) + 1
        largest_region = labeled_image == largest_label

        # calculate the position of the largest region
        position = [coords.mean() for coords in  np.where(largest_region)]

        # find the maximum extent in the x and y directions
        x_extent = np.sum(largest_region, axis=0).max()
        y_extent = np.sum(largest_region, axis=1).max()

        # return the average of the two extents in micrometres
        return (x_extent + y_extent) / 2 * self.pixel_size * 1e6, tuple(position)

    def measure_xy_diameters(self) -> list:
        """Measure the diameters of all connected regions in the image.
        
        Returns:
            list: The list of diameters in micrometres.
        """
        # threshold the image at 50% of the initial intensity
        thresholded_image = self < 0.5

        # iterate over connected regions
        labeled_image, n_labels = ndimage.label(thresholded_image, structure=np.ones((3, 3)))

        diameters = []
        for label in range(1, n_labels + 1):
            region = labeled_image == label

            # find the maximum extent in the x and y directions
            x_extent = np.sum(region, axis=0).max()
            y_extent = np.sum(region, axis=1).max()

            diameters.append((x_extent + y_extent) / 2 * self.pixel_size * 1e6)

        # return list of diameters in micrometres
        return diameters

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

        n_pixels = np.sum((self <= max_intensity_counted)
                          & (self > min_intensity_counted))

        return n_pixels


@dataclass
class ASTModel:
    r"""Angular Spectrum Theory model.

    A model to contain the diffraction behaviour of a single opaque object,
    exposed to a plane wave of light at a specific wavelength, and imaged
    using a specific pixel size.

    Args:
        opaque_shape (np.ndarray): The shape of the opaque object in the z=0 plane.
        wavenumber (float, optional): The wavenumber of the light. Defaults to :math:`2\pi / \text{(658 nm)}`.
        pixel_size (float, optional): The size of the pixels in the opaque_shape. Defaults to 10 µm.
    """

    opaque_shape: np.ndarray
    wavenumber: float = 2 * np.pi / 658e-9  # in inverse metres
    pixel_size: float = 10e-6  # in metres

    def __post_init__(self):
        self.amplitudes = {}  # z: phase grid
        self.diameters = {}

        # trim opaque shape of zero-valued rows and columns
        nonzero_x = np.arange(
            self.opaque_shape.shape[0])[self.opaque_shape.any(axis=1)]
        nonzero_y = np.arange(
            self.opaque_shape.shape[1])[self.opaque_shape.any(axis=0)]
        if nonzero_x.size > 0:
            self.opaque_shape = self.opaque_shape[
                nonzero_x.min():nonzero_x.max() + 1, :]
        if nonzero_y.size > 0:
            self.opaque_shape = self.opaque_shape[:,
                                                  nonzero_y.min(
                                                  ):nonzero_y.max() + 1]

    @classmethod
    def from_diameter(cls, diameter: float, wavenumber:float=None, pixel_size:float=None):
        r"""Create a model for a circular opaque object.

        Args:
            diameter (float): The diameter of the opaque object in micrometres.
            wavenumber (float, optional): The wavenumber of the light. Defaults to :math:`2\pi / \text{(658 nm)}`.
            pixel_size (float, optional): The size of the pixels in the opaque_shape. Defaults to 10 µm.

        Returns:
            ASTModel: The model.
        """
        # set defaults
        if wavenumber is None:
            wavenumber = cls.wavenumber
        if pixel_size is None:
            pixel_size = cls.pixel_size

        # create the opaque shape
        radius_m = diameter * 1e-6 / 2
        radius_px = radius_m / pixel_size
        x = np.arange(-radius_px, radius_px)
        y = np.arange(-radius_px, radius_px)
        x_val_grid, y_val_grid = np.meshgrid(x, y)
        opaque_shape = np.where(x_val_grid**2 + y_val_grid**2 <= radius_px**2, 1, 0)

        # create the model
        model = cls(opaque_shape, wavenumber, pixel_size)
        model.diameters["true"] = diameter * 1e-6

        return model
    
    @classmethod
    def from_rectangle(cls, width: float, height: float, angle: float=0, wavenumber: float=None, pixel_size: float=None):
        r"""Create a model for a rectangular opaque object.
        
        Args:
            width (float): The width of the opaque object in micrometres.
            height (float): The height of the opaque object in micrometres.
            angle (float): The angle of the opaque object in degrees.
            wavenumber (float, optional): The wavenumber of the light. Defaults to :math:`2\pi / \text{(658 nm)}`.
            pixel_size (float, optional): The size of the pixels in the opaque_shape. Defaults to 10 µm."""
        
        # set defaults
        if wavenumber is None:
            wavenumber = cls.wavenumber
        if pixel_size is None:
            pixel_size = cls.pixel_size
        
        # create the opaque shape
        height_px = int(height * 1e-6 // pixel_size)
        width_px = int(width * 1e-6 // pixel_size)
        opaque_shape = np.ones((height_px, width_px))
        
        # rotate the opaque shape
        opaque_shape = ndimage.rotate(opaque_shape, angle)
        
        # create the model
        model = cls(opaque_shape, wavenumber, pixel_size)
        model.diameters["true"] = width * 1e-6
        
        return model



    def process(self, z_val: float, low_pass=1.0) -> AmplitudeField:
        """Process the model, calculating the amplitude given the opaque shape is at z.

        Args:
            z_val (float): The distance of the the opaque_shape from the object plane.
            low_pass (float, optional): The low pass filter to apply to the amplitude. Defaults to 1.0.

        Returns:
            AmplitudeField: The amplitude (transmission function) of the image at z.
        """

        # check if the amplitude has already been calculated
        if z_val in self.amplitudes:
            return self.amplitudes[z_val]

        object_plane = np.pad(
            self.opaque_shape,
            max(self.opaque_shape.shape) *
            10,  # arbitrarily 10 times the size of the object
            "constant",
            constant_values=(0, 0),
        )

        # calculate the transmission function (1 outside the shape, 0 inside)
        transmission_function = np.where(object_plane, 0, 1)

        # transform into fourier space
        transmission_function_fourier = np.fft.fft2(transmission_function)

        # calculate the fourier space coordinates
        f_x = np.fft.fftfreq(transmission_function.shape[1], self.pixel_size)
        f_y = np.fft.fftfreq(transmission_function.shape[0],
                             self.pixel_size).reshape(-1, 1)

        # low pass filter
        # FIXME: I have no idea if this works/is the right vibe.
        f_xy = np.meshgrid(f_x, f_y)
        transmission_function_fourier = np.where(
            f_xy[0]**2 + f_xy[1]**2
            <= (f_xy[0]**2 + f_xy[1]**2).max() * low_pass,
            transmission_function_fourier,
            0,
        )

        # apply helmholtz phase factor
        helmholtz_phase_factor = np.sqrt(self.wavenumber**2 - 4 * np.pi**2 *
                                         (f_x**2 + f_y**2)
                                         )
        transmission_function_fourier_translated = (
            transmission_function_fourier *
            np.exp(1j * z_val * helmholtz_phase_factor))

        # transform back into real space
        transmission_function_translated = np.fft.ifft2(
            transmission_function_fourier_translated)
        
        amplitude = AmplitudeField(transmission_function_translated, pixel_size=self.pixel_size)

        self.amplitudes[z_val] = amplitude
        return amplitude

    def process_range(self, z_range: np.ndarray) -> np.ndarray:
        """Process the model for offsetting the object at range of z values.

        Args:
            z_range (np.ndarray): The range of z values to process.

        Returns:
            np.ndarray: The intensity of the image at z.
        """
        return [self.process(z) for z in z_range]

    def plot_intensity(self, z_val: int, **kwargs) -> plt.Axes:
        """Plot the intensity at a given z

        Args:
            z (float): The distance of the the opaque_shape from the object plane.
        """
        amplitude = self.process(z_val)

        return amplitude.intensity.plot(**kwargs)

    def xy_diameter(self) -> float:
        """Calculate the xy diameter of the opaque_shape."""
        if self.diameters.get("xy"):
            return self.diameters["xy"]

        nonzero_x = np.arange(
            self.opaque_shape.shape[0])[self.opaque_shape.any(axis=1)]
        nonzero_y = np.arange(
            self.opaque_shape.shape[1])[self.opaque_shape.any(axis=0)]

        self.diameters["xy"] = (np.mean([
            nonzero_x.max() - nonzero_x.min(),
            nonzero_y.max() - nonzero_y.min()
        ]) * self.pixel_size)
        return self.diameters["xy"]

    def get_zd(self, z_val: float, diameter_type: str) -> float:
        r"""Calculate the dimensionless diffraction z distance.

        The diffraction pattern for a particle (sized relative to the true diameter)
        is a function of the dimensionless diffraction z distance, 

        .. math :: 

            z_\text{D} = \frac{4 \pi z}{D^2}.
        
        Args:
            z_val (float): The distance of the the opaque_shape from the object plane.
            diameter_type (str): The type of diameter to use.
            
        Returns:
            float: The dimensionless diffraction z distance."""
        wavelength = 2 * np.pi / self.wavenumber
        return 4 * wavelength * z_val / self.diameters[diameter_type]**2
    
    def rescale(self, diameter_scale_factor):
        """Produce a new AST model for similar object at a different scale.

        Scales the object, and any already-measured diameters or processed amplitude 
        profiles by a given factor.

        Args:
            diameter_scale_factor (float): The factor by which to scale the object's diameter.

        Returns:
            ASTModel: The new AST model object.
        """
        # create a copy of the original object
        scaled_model = deepcopy(self)

        # scale the object by altering the pixel size
        scaled_model.pixel_size = self.pixel_size * diameter_scale_factor

        # scale existing diameters 
        for diameter_type, value in scaled_model.diameters.items():
            scaled_model.diameters[diameter_type] = value * diameter_scale_factor
        # TODO: Store diameters as pixel units so that this is not required?

        # scale the z value at which the diffraction patterns occur
        # z is proportional to D^1/2
        scaled_model.amplitudes = {}
        for z_val, amplitude_profile in self.amplitudes.items():
            scaled_z_val = z_val * diameter_scale_factor**0.5 
            # TODO: make sure cached z values here have a specified precision, so they're actually reused.
            scaled_model.amplitudes[scaled_z_val] = deepcopy(amplitude_profile)
            scaled_model.amplitudes[scaled_z_val].pixel_size = scaled_model.pixel_size

        # return the new ASTModel object
        return scaled_model
    
    def regrid(self, pixel_size= 10e-6):
        """Regrid the model to a new pixel size.

        Resamples the opaque object and any already-measured amplitude profiles 
        to a new pixel size, such as one that matches the real detector.
        
        Args:
            pixel_size (float): The pixel size to move to.
        """
        scale_factor = pixel_size / self.pixel_size

        self.opaque_shape = rescale(self.opaque_shape, 1/scale_factor)
        self.pixel_size = pixel_size

        for z_val, amplitude_profile in self.amplitudes.items():
            #FIXME: this doesn't work, because amplitudes are complex
            # self.amplitudes[z_val] = AmplitudeField(rescale(amplitude_profile, 1/scale_factor), pixel_size=pixel_size)
            pass