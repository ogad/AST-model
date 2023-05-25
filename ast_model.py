# Angular Spectrum Theory model
# Authour: Oliver Driver
# Date: 22/05/2023

# Standard library imports
from dataclasses import dataclass

# Package imports
import numpy as np
from matplotlib import pyplot as plt


class IntensityField(np.ndarray):
    """A class to represent an intensity field

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

    def plot(self, ax=None, axis_length=None, grayscale_bounds=None, **kwargs):
        """Plot the intensity field

        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            axis_length(float): The axis_length of the object in micrometres.
            grayscale_bounds (list): The list of bounds for grayscale bands. When None, the intensity is plotted.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if axis_length is not None:
            axis_length_px = axis_length * 1e-6 // self.pixel_size
            to_plot = self[
                int((self.shape[0] - axis_length_px) // 2) : int(
                    (self.shape[0] + axis_length_px) // 2
                ),
                int((self.shape[1] - axis_length_px) // 2) : int(
                    (self.shape[1] + axis_length_px) // 2
                ),
            ]
        else:
            to_plot = self

        if grayscale_bounds is not None:
            # Replace pixel values to the next-highest grayscale bound
            bounded = np.zeros_like(to_plot)
            for i, bound in enumerate(sorted(grayscale_bounds)):
                current_bound = (to_plot < bound) & (bounded == 0)
                to_plot = np.where(current_bound, i, to_plot)
                bounded = np.where(current_bound, 1, bounded)
            to_plot = np.where((bounded == 0), len(grayscale_bounds), to_plot)

        ax_image = ax.imshow(to_plot, **kwargs)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        ax.set_xticklabels((xticks * self.pixel_size * 1e6).round(1))
        ax.set_yticklabels((yticks * self.pixel_size * 1e6).round(1))
        ax.set_xlabel("x/µm")
        ax.set_ylabel("y/µm")

        return ax_image

    def n_pixels_depletion_range(self, min_dep, max_dep):
        """Calculate the number of pixels in the depletion range

        Args:
            min_dep (float): The minimum depletion value.
            max_dep (float): The maximum depletion value.

        Returns:
            int: The number of pixels in the depletion range.
        """
        min_intensity_counted = 1 - max_dep
        max_intensity_counted = 1 - min_dep

        n_pixels = np.sum(
            (self <= max_intensity_counted) & (self > min_intensity_counted)
        )

        return n_pixels


@dataclass
class ASTModel:
    """Angular Spectrum Theory model

    Args:
        opaque_shape (np.ndarray): The shape of the opaque object in the z=0 plane.
        wavenumber (float, optional): The wavenumber of the light. Defaults to 2 * np.pi / (658 nm).
        pixel_size (float, optional): The size of the pixels in the opaque_shape. Defaults to 10 µm.
    """

    opaque_shape: np.ndarray
    wavenumber: float = 2 * np.pi / 658e-9  # in inverse metres
    pixel_size: float = 10e-6  # in metres

    def __post_init__(self):
        self.intenisties = {}  # z: intenisty grid
        self.diameters = {}

        # trim opaque shape of zero-valued rows and columns
        nonzero_x = np.arange(self.opaque_shape.shape[0])[self.opaque_shape.any(axis=1)]
        nonzero_y = np.arange(self.opaque_shape.shape[1])[self.opaque_shape.any(axis=0)]
        self.opaque_shape = self.opaque_shape[
            nonzero_x.min() : nonzero_x.max() + 1, nonzero_y.min() : nonzero_y.max() + 1
        ]

    @classmethod
    def from_diameter(cls, diameter, wavenumber=None, pixel_size=None):
        """Create a model for a circular opaque object.

        Args:
            diameter (float): The diameter of the opaque object in micrometres.
            wavenumber (float, optional): The wavenumber of the light. Defaults to 2 * np.pi / (658 nm).
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
        xx, yy = np.meshgrid(x, y)
        opaque_shape = np.where(xx**2 + yy**2 <= radius_px**2, 1, 0)

        # create the model
        model = cls(opaque_shape, wavenumber, pixel_size)
        model.diameters["true"] = diameter * 1e-6

        return model

    def process(self, z_val: int, low_pass=1.0) -> IntensityField:
        """Process the model for a given z

        Args:
            z (float): The distance of the the opaque_shape from the object plane.

        Returns:
            IntensityField: The intensity of the image at z.
        """

        # check if the intensity has already been calculated
        if z_val in self.intenisties:
            return self.intenisties[z_val]

        object_plane = np.pad(
            self.opaque_shape,
            max(self.opaque_shape.shape)
            * 10,  # arbitrarily 10 times the size of the object
            "constant",
            constant_values=(0, 0),
        )

        # calculate the transmission function (1 outside the shape, 0 inside)
        transmission_function = np.where(object_plane, 0, 1)

        # transform into fourier space
        transmission_function_fourier = np.fft.fft2(transmission_function)

        # calculate the fourier space coordinates
        f_x = np.fft.fftfreq(transmission_function.shape[1], self.pixel_size)
        f_y = np.fft.fftfreq(transmission_function.shape[0], self.pixel_size).reshape(
            -1, 1
        )

        # low pass filter
        # f_xy = np.meshgrid(f_x, f_y)
        # transmission_function_fourier = np.where(
        #     f_xy[0] ** 2 + f_xy[1] ** 2 <= f_x.max() * low_pass,
        #     transmission_function_fourier,
        #     0,
        # )

        # apply helmholtz phase factor
        helmholtz_phase_factor = np.sqrt(
            self.wavenumber**2 - 4 * np.pi**2 * (f_x**2 + f_y**2)
        )
        transmission_function_fourier_translated = (
            transmission_function_fourier * np.exp(1j * z_val * helmholtz_phase_factor)
        )

        # transform back into real space
        transmission_function_translated = np.fft.ifft2(
            transmission_function_fourier_translated
        )

        # calculate the intensity
        intensity_translated = np.abs(transmission_function_translated) ** 2
        intensity_translated_as_field = IntensityField(
            intensity_translated, pixel_size=self.pixel_size
        )

        # cache and return the intensity
        self.intenisties[z_val] = intensity_translated_as_field
        return intensity_translated_as_field

    def process_range(self, z_range: np.ndarray) -> np.ndarray:
        """Process the model for a range of z values

        Args:
            z_range (np.ndarray): The range of z values to process.

        Returns:
            np.ndarray: The intensity of the image at z.
        """
        return [self.process(z) for z in z_range]

    def plot_intensity(self, z_val: int, **kwargs):
        """Plot the intensity at a given z

        Args:
            z (float): The distance of the the opaque_shape from the object plane.
        """
        intensity = self.process(z_val)

        return intensity.plot(**kwargs)

    def xy_diameter(self):
        """Calculate the xy diameter of the opaque_shape, defined to be the mean of the maximum nonzero extent of the opaque shape in the x and y directions."""
        if self.diameters.get("xy"):
            return self.diameters["xy"]

        nonzero_x = np.arange(self.opaque_shape.shape[0])[self.opaque_shape.any(axis=1)]
        nonzero_y = np.arange(self.opaque_shape.shape[1])[self.opaque_shape.any(axis=0)]

        self.diameters["xy"] = (
            np.mean(
                [nonzero_x.max() - nonzero_x.min(), nonzero_y.max() - nonzero_y.min()]
            )
            * self.pixel_size
        )
        return self.diameters["xy"]

    def get_zd(self, z_val, diameter_type):
        """Calculate the dimensionless diffraction z distance."""
        wavelength = 2 * np.pi / self.wavenumber
        return 4 * wavelength * z_val / self.diameters[diameter_type] ** 2
