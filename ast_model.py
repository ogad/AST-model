# Angular Spectrum Theory model
# Authour: Oliver Driver
# Date: 22/05/2023

# Standard library imports
from dataclasses import dataclass

# Package imports
import numpy as np


@dataclass
class ASTModel:
    """Angular Spectrum Theory model

    Args:
        opaque_shape (np.ndarray): The shape of the opaque object in the z=0 plane.
        wavenumber (float, optional): The wavenumber of the light. Defaults to 2 * np.pi / (658 nm).
        pizel_size (float, optional): The size of the pixels in the opaque_shape. Defaults to 10 µm.
    """

    opaque_shape: np.ndarray
    wavenumber: float = 2 * np.pi / 658e-9  # in inverse metres
    pizel_size: float = 10e-6  # in metres

    def process(self, z_val: int) -> np.ndarray:
        """Process the model for a given z

        Args:
            z (float): The distance of the the opaque_shape from the object plane.

        Returns:
            np.ndarray: The intensity of the image at z.
        """

        # calculate the transmission function (1 outside the shape, 0 inside)
        transmission_function = np.where(self.opaque_shape, 0, 1)

        # transform into fourier space
        transmission_function_fourier = np.fft.fft2(transmission_function)

        # calculate the fourier space coordinates
        f_x = np.fft.fftfreq(transmission_function.shape[0], self.pizel_size)
        f_y = np.fft.fftfreq(transmission_function.shape[1], self.pizel_size).reshape(
            -1, 1
        )

        # apply helmholtz phase factor
        helmholtz_phase_factor = np.sqrt(
            self.wavenumber**2 - 4 * np.pi * (f_x**2 + f_y**2)
        )

        transmission_function_fourier_translated = (
            transmission_function_fourier * np.exp(1j * z_val * helmholtz_phase_factor)
        )

        # transform back into real space
        transmission_function_translated = np.fft.ifft2(
            transmission_function_fourier_translated
        )

        # return the intensity
        return np.abs(transmission_function_translated) ** 2

    def process_range(self, z_range: np.ndarray) -> np.ndarray:
        """Process the model for a range of z values

        Args:
            z_range (np.ndarray): The range of z values to process.

        Returns:
            np.ndarray: The intensity of the image at z.
        """
        return np.array([self.process(z) for z in z_range])
