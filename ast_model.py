# Angular Spectrum Theory model
# Authour: Oliver Driver
# Date: 22/05/2023

from dataclasses import dataclass

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

    def process(self, z: float):
        """Process the model

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
        transmission_function_fourier_translated = (
            transmission_function_fourier
            * np.exp(
                1j
                * z
                * np.sqrt(self.wavenumber**2 - 4 * np.pi * (f_x**2 + f_y**2))
            )
        )

        # transform back into real space
        transmission_function_translated = np.fft.ifft2(
            transmission_function_fourier_translated
        )

        # return the intensity
        return np.abs(transmission_function_translated) ** 2


if __name__ == "__main__":
    # test the model
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # define the object
    image = np.zeros((16, 16))
    image[7:9, 7:9] = 1

    # create the model
    model = ASTModel(opaque_shape=object)

    # calculate the intensity at different z values
    z = 2e-3  # in metres
    intensity = model.process(z)

    # plot the intensity
    intensity_norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(1 - image, norm=intensity_norm, cmap="BrBG_r")
    ax1.set_title("Object at z = 0mm")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.imshow(intensity, norm=intensity_norm, cmap="BrBG_r")
    ax2.set_title(f"Intensity at z = {z*1000}mm")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.tight_layout()

    plt.show()
