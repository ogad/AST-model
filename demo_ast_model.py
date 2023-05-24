# %%

# Package imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from imageio import imread

# Local imports
from ast_model import ASTModel

# define the object

## square object
# image = np.zeros((16, 16))
# image[7:9, 7:9] = 1

## basic rosette image
# im = imread("rosette_basic.png")
# image = np.where(im.max(axis=2), 1, 0)

## circle object
radius = 6  # in 10s of microns
image = np.zeros((16, 16))
for xx in range(16):
    for yy in range(16):
        if (xx - 8) ** 2 + (yy - 8) ** 2 < radius**2:
            image[xx, yy] = 1
image_shape = image.shape

expanded_image = np.zeros((512, 512))
# place image in the centre of the expanded image
start_image_x = 256 - image_shape[0] // 2
start_image_y = 256 - image_shape[1] // 2
expanded_image[
    start_image_x : start_image_x + image_shape[0],
    start_image_y : start_image_y + image_shape[1],
] = image

# create the model
model = ASTModel(opaque_shape=expanded_image)

# calculate the intensity at different z values
z = 2e-3  # in metres
intensity = model.process(z)

# Plotting
## Define colourmap normalisation
intensity_norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)

## Create figure
fig, (ax1, ax2) = plt.subplots(1, 2)

## Plot the object on first axis
ax1.imshow(1 - image, norm=intensity_norm, cmap="BrBG_r")
ax1.set_title("Object at z = 0mm")
ax1.set_xlabel("x/10µm")
ax1.set_ylabel("y/10µm")

## Plot the diffracted intensity on second axis
intensity_unextended = intensity[
    start_image_x : start_image_x + image_shape[0],
    start_image_y : start_image_y + image_shape[1],
]
# a = ax2.imshow(intensity_unextended, norm=intensity_norm, cmap="BrBG_r")
a = intensity_unextended.plot(ax=ax2, norm=intensity_norm, cmap="BrBG_r")
ax2.set_title(f"Intensity at z = {z*1000}mm")
ax2.set_xlabel("x/10µm")
ax2.set_ylabel("y/10µm")

ax3 = fig.add_subplot(8, 1, 8)
fig.colorbar(a, cax=ax3, orientation="horizontal")
ax3.set_title("Intensity / $I_0$")

plt.tight_layout()

plt.show()

# %%
z_values = np.arange(-150, 151, 0.2)  # in millimetres
zd_values = 2 * 658e-9 * z_values * 1e-3 / (radius * 10e-6) ** 2
a = model.process_range(z_values * 1e-3)
plt.plot(
    zd_values,
    [
        arr.n_pixels_depletion_range(0.25, 0.5)
        / arr.n_pixels_depletion_range(0.25, 1.0)
        for arr in a
    ],
)

# %%
