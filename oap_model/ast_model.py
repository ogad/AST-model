
import numpy as np
from affine import Affine
from matplotlib import pyplot as plt
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import Point
from skimage.transform import rescale


from copy import deepcopy
from dataclasses import dataclass

from .intensity import AmplitudeField

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
        self.wavelength = 2 * np.pi / self.wavenumber

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
        # x = np.arange(-radius_px, radius_px)
        # y = np.arange(-radius_px, radius_px)
        # x_val_grid, y_val_grid = np.meshgrid(x, y)
        # opaque_shape = np.where(x_val_grid**2 + y_val_grid**2 <= radius_px**2, 1, 0)

        particle_shapely = Point(0,0).buffer(radius_m)
        output_len = int(2*radius_px + 5)
        if radius_m == 0:
            opaque_shape = np.zeros((1,1))
        else:
            opaque_shape = rasterize([particle_shapely], out_shape=(output_len,output_len), transform=Affine.scale(pixel_size) * Affine.translation(-(output_len+1)/2, -(output_len+1)/2)) #place centre on a vertex not a pixel centre


        # create the model
        model = cls(opaque_shape, wavenumber, pixel_size)
        model.diameters["true"] = diameter * 1e-6

        return model

    @staticmethod
    def rectangle(px_height, px_width, angle:float|tuple[float,float]=0.):
        r"""Create a rectangular opaque object.

        Args:
            px_height (int): The height of the opaque object in pixels.
            px_width (int): The width of the opaque object in pixels.
            angle (float|tuple[float,float], optional): The angle of the opaque object in degrees. Defaults to 0.

        Returns:
            np.ndarray: The shape of the opaque object.
        """
        # if a tuple of angles are given, rotate in 3D.
        if isinstance(angle, tuple):
            angle_in_plane = angle[0]
            angle_out_of_plane = angle[1]
        else:
            angle_in_plane = angle
            angle_out_of_plane = 0.

        # rotation approximated by shortening the longer side
        if angle_out_of_plane != 0.:
            if px_height > px_width:
                px_width = px_width * np.cos(angle_out_of_plane)
            else:
                px_height = px_height * np.cos(angle_out_of_plane)

        # create the opaque shape
        px_height = int(round(px_height))
        px_width = int(round(px_width))

        shape = np.ones((px_width, px_height))

        # rotate the shape
        shape = ndimage.rotate(shape, angle_in_plane * 180/np.pi)

        return shape

    @classmethod
    def from_rectangle(cls, width: float, height: float, angle: float|tuple[float,float]=0, wavenumber: float=None, pixel_size: float=None):
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
            pixel_size = cls.pixel_size #TODO: this should inherit from a detector!

        # create the opaque shape
        height_px = height * 1e-6 / pixel_size
        width_px = width * 1e-6 / pixel_size

        opaque_shape = cls.rectangle(height_px, width_px, angle)
        if opaque_shape.size == 0:
            opaque_shape = np.zeros((1,1))

        # create the model
        model = cls(opaque_shape, wavenumber, pixel_size)
        model.diameters["true"] = (4*sum(opaque_shape) * pixel_size**2 / np.pi)**0.5

        return model

    @classmethod
    def from_diameter_rectangular(cls, diameter: float, aspect_ratio: float, **kwargs):
        """Create a model for a rectangular opaque object with a given diameter xy mean diameter and aspect ratio."""
        area = np.pi * (diameter / 2)**2
        width = np.sqrt(area / aspect_ratio)
        height = width * aspect_ratio
        # width = 2 * diameter / (1 + aspect_ratio)
        # height = width * aspect_ratio

        return cls.from_rectangle(width, height, **kwargs)

    @classmethod
    def from_diameter_rosette(cls, diameter:float, n_rectangles:int, width:float=40, wavenumber:float=None, pixel_size:float=None, **kwargs):
        """Create a model for a rosette opaque object made up of n_rectangles opaque objects."""
        # set defaults
        if wavenumber is None:
            wavenumber = cls.wavenumber
        if pixel_size is None:
            pixel_size = cls.pixel_size


        if diameter <= width:
            px_length = diameter*1e-6 / pixel_size
        else:
            approx_overlap_area = width **2
            target_area = np.pi * (diameter / 2)**2
            length = np.round((target_area + (n_rectangles-1)*approx_overlap_area) / (n_rectangles * width))
            px_length = length*1e-6 / pixel_size

        px_width = width*1e-6 / pixel_size

        # create the opaque shape
        angles = np.linspace(0, np.pi, n_rectangles, endpoint=False)
        opaque_shapes = [cls.rectangle(px_length, px_width, angle) for angle in angles]

        # work out the size of the opaque shape
        opaque_shape_size = np.max([shape.shape for shape in opaque_shapes], axis=0)
        # create the opaque shape
        opaque_shape = np.zeros(opaque_shape_size)
        for shape in opaque_shapes:
            x_offset = int((opaque_shape_size[0] - shape.shape[0]) / 2)
            y_offset = int((opaque_shape_size[1] - shape.shape[1]) / 2)
            opaque_shape[x_offset:x_offset+shape.shape[0], y_offset:y_offset+shape.shape[1]] += shape

        # flatten opaque shape
        opaque_shape = np.where(opaque_shape > 0, 1, 0)

        # create the model
        model = cls(opaque_shape, wavenumber, pixel_size)
        model.diameters["true"] = diameter * 1e-6

        return model


    def process(self, z_val: float, low_pass=1.) -> AmplitudeField:
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

        diameter_overestimate = (max(self.opaque_shape.shape)*self.pixel_size)
        estimated_airy_diameter_px = int(1.22 * self.wavelength * z_val/diameter_overestimate / self.pixel_size)
        if abs(estimated_airy_diameter_px) > 20:
            # costly to calculate, so only do it if it's going to be used
            theoretical_dof = 8*diameter_overestimate**2 / (self.wavelength * 4)
            if abs(z_val) > 10*theoretical_dof:
                return AmplitudeField(np.ones_like(self.opaque_shape), pixel_size=self.pixel_size)

        if self.opaque_shape.size == 0:
            object_plane = np.zeros((1, 1))
        else:
            object_plane = np.pad(
                self.opaque_shape,
                # int(2*((max(self.opaque_shape.shape)*self.pixel_size)**2 / (4*self.wavelength)) // self.pixel_size),
                max(
                    abs(10 * estimated_airy_diameter_px),
                    10),
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
        # f_xy = np.meshgrid(f_x, f_y)
        # transmission_function_fourier = np.where(
        #     f_xy[0]**2 + f_xy[1]**2
        #     <= (f_xy[0]**2 + f_xy[1]**2).max() * low_pass,
        #     transmission_function_fourier,
        #     0,
        # )

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