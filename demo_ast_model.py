# %%

# Package imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from imageio.v3 import imread

# Local imports
from ast_model import ASTModel

# define the object

## square object
# image = np.ones((8, 8))
# image[7:9, 7:9] = 1

## basic rosette image
im = imread("rosette_basic.png")
image = np.where(im.max(axis=2), 1, 0)

# # create the model
model = ASTModel(opaque_shape=image)
# model = ASTModel.from_diameter(150, pixel_size=5e-6)

# calculate the intensity at different z values
z = 5e-3  # in metres

# Plotting
## Define colourmap normalisation
intensity_norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)

## Create figure
fig, (ax1, ax2) = plt.subplots(1, 2)

## Plot the object on first axis
model.plot_intensity(0, ax=ax1, axis_length=400, norm=intensity_norm, cmap="BrBG_r")
ax1.set_title("Object at z = 0mm")

## Plot the diffracted intensity on second axis
# a = ax2.imshow(intensity_unextended, norm=intensity_norm, cmap="BrBG_r")
# a = model.plot_intensity(z, ax=ax2, axis_length=300, norm=intensity_norm, cmap="BrBG_r")
a = model.plot_intensity(
    z, ax=ax2, axis_length=400, grayscale_bounds=[0.25, 0.5, 0.75, 1]
)
ax2.set_title(f"Intensity at z = {z*1000}mm")

ax3 = fig.add_subplot(8, 1, 8)
fig.colorbar(a, cax=ax3, orientation="horizontal")
ax3.set_title("Intensity / $I_0$")

plt.tight_layout()

plt.show()


# %% Reproducing ratio functions (O'Shea 2019, Figure 7)
def plot_grayscale_ratio(range1, range2, ax):
    z_values = np.arange(-0.150, 0.151, 0.001)  # in m
    zd_values = model.get_zd(z_values, "true")
    a = model.process_range(z_values)
    ax.plot(
        zd_values,
        # z_values * 1e3,
        [
            arr.n_pixels_depletion_range(*range1)
            / arr.n_pixels_depletion_range(*range2)
            for arr in a
        ],
    )

    # ax.set_xlabel("z (mm)")
    ax.set_xlabel("$z_d$")
    ax.set_ylabel(
        f"$A_{{{int(100*range1[0])}-{int(100*range1[1])}}}/A_{{{int(100*range2[0])}-{int(100*range2[1])}}}$"
    )

    ax.grid(True)
    ax.set_xlim(-10, 10)
    # ax.set_ylim(0, 1)
    # ax.set_xlim(-500, 500)


ratios = [
    ((0.25, 0.5), (0.25, 1.0)),
    ((0.75, 1.0), (0.5, 0.75)),
    ((0.5, 0.75), (0.25, 1.0)),
    ((0.5, 0.75), (0.25, 0.5)),
    ((0.75, 1.0), (0.25, 1.0)),
    ((0.75, 1.0), (0.25, 0.5)),
]

fig, axs = plt.subplots(3, 2, figsize=(8, 10))
for i, ratio in enumerate(ratios):
    plot_grayscale_ratio(*ratio, axs.reshape(-1)[i])

plt.tight_layout()
plt.show()

# %%
