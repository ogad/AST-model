# %%
# Package imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from imageio.v3 import imread

# Local imports
from oap_model.ast_model import ASTModel

if __name__ == "__main__":
    # define the object

    ## square object
    # image = np.ones((8, 8))
    # image[7:9, 7:9] = 1

    ## basic rosette image
    im = imread("rosette_basic.png")
    image = np.where(im.max(axis=2), 1, 0)

    # # create the model
    # model = ASTModel(opaque_shape=image)
    model = ASTModel.from_diameter(150, pixel_size=5e-6)

    # calculate the intensity at different z values
    z_values = [0, 5e-3, 10e-3, 20e-3]  # in metres

    # Plotting
    ## Define colourmap normalisation
    intensity_norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=2)

    ## Create figure
    # fig, axs = plt.subplots(3, 2, figsize=(6, 8), height_ratios=[1, 1, 0.2])
    fig = plt.figure(figsize=(8, 10))
    ax0 = plt.subplot2grid((17, 2), (0, 0), rowspan=8)
    ax1 = plt.subplot2grid((17, 2), (0, 1), rowspan=8)
    ax2 = plt.subplot2grid((17, 2), (8, 0), rowspan=8)
    ax3 = plt.subplot2grid((17, 2), (8, 1), rowspan=8)
    cax = plt.subplot2grid((17, 2), (16, 0), colspan=2)
    axs = np.array([[ax0, ax1], [ax2, ax3]])

    ## Plot the object on first axis
    for i, z_value in enumerate(z_values):
        ax = axs.reshape(-1)[i]
        a = model.plot_intensity(
            z_value, ax=ax, axis_length=400, norm=intensity_norm, cmap="BrBG_r"
        )
        ax.set_title(f"Intensity at z = {int(z_value*100)}mm")

    fig.colorbar(a, cax=cax, orientation="horizontal")
    cax.set_title("Intensity / $I_0$")

    fig.tight_layout()

# %%
if __name__ == "__main__":
    # Plot in the grayscale bins
    ## Create figure
    # fig, axs = plt.subplots(3, 2, figsize=(6, 8), height_ratios=[1, 1, 0.2])
    fig = plt.figure(figsize=(8, 10))
    ax0 = plt.subplot2grid((17, 2), (0, 0), rowspan=8)
    ax1 = plt.subplot2grid((17, 2), (0, 1), rowspan=8)
    ax2 = plt.subplot2grid((17, 2), (8, 0), rowspan=8)
    ax3 = plt.subplot2grid((17, 2), (8, 1), rowspan=8)
    cax = plt.subplot2grid((17, 2), (16, 0), colspan=2)
    axs = np.array([[ax0, ax1], [ax2, ax3]])

    ## Plot the object on first axis
    for i, z_value in enumerate(z_values):
        ax = axs.reshape(-1)[i]
        a = model.plot_intensity(
            z_value,
            ax=ax,
            axis_length=400,
            grayscale_bounds=[0.25, 0.5, 0.75],
            cmap="Greys_r",
        )
        ax.set_title(f"Intensity at z = {int(z_value*100)}mm")

    fig.colorbar(a, cax=cax, orientation="horizontal")
    cax.set_title("Intensity / $I_0$")

    fig.tight_layout()

    # %% Reproducing ratio functions for droplet (O'Shea 2019, Figure 7)
    # When a "true" diameter has been given to the model, we can convert to zd coordinates,
    # when not, be sure to switch out the line in the plot function
    model = ASTModel.from_diameter(150, pixel_size=5e-6)


    def plot_grayscale_ratio_droplet(range1, range2, ax):
        z_values = np.arange(-0.150, 0.151, 0.001)  # in m
        zd_values = model.get_zd(z_values, "true")
        a = model.process_range(z_values)
        ax.plot(
            zd_values,
            [
                arr.intensity.n_pixels_depletion_range(*range1)
                / arr.intensity.n_pixels_depletion_range(*range2)
                for arr in a
            ],
        )

        ax.set_xlabel("z (mm)")
        # ax.set_xlabel("$z_d$")
        ax.set_ylabel(
            f"$A_{{{int(100*range1[0])}-{int(100*range1[1])}}}/A_{{{int(100*range2[0])}-{int(100*range2[1])}}}$"
        )

        ax.grid(True)
        ax.set_xlim(-10, 10)


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
        plot_grayscale_ratio_droplet(*ratio, axs.reshape(-1)[i])

    plt.show()

    # %%
    # %%
    # Reproducing ratio functions for rosette (O'Shea 2021, Figure 5)

    # When a "true" diameter has been given to the model, we can convert to zd coordinates,
    # when not, be sure to switch out the line in the plot function
    model = ASTModel(opaque_shape=image)
    # model.diameters["true"] = 150e-6


    def plot_grayscale_ratio_rosette(range1, range2=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        z_values = np.arange(-0.150, 0.151, 0.001)  # in m
        model.diameters["true"] = 118e-6
        zd_values = model.get_zd(z_values, "true")
        a = model.process_range(z_values)

        if range2:
            ax.plot(
                zd_values,
                # z_values * 1e3,
                [
                    arr.intensity.n_pixels_depletion_range(*range1)
                    / arr.intensity.n_pixels_depletion_range(*range2)
                    for arr in a
                ],
            )
            ax.set_ylabel(
                f"$A_{{{int(100*range1[0])}-{int(100*range1[1])}}}/A_{{{int(100*range2[0])}-{int(100*range2[1])}}}$"
            )
        else:
            ax.plot(
                zd_values,
                # z_values * 1e3,
                [arr.intensity.n_pixels_depletion_range(*range1) * (10 / 15) ** 2 for arr in a],
            )
            ax.set_ylabel(f"$A_{{{int(100*range1[0])}-{int(100*range1[1])}}}$")

        ax.set_xlabel("z (mm)")
        # ax.set_xlabel("$z_d$")

        ax.grid(True)
        ax.set_xlim(-50, 50)


    ratios = [
        ((0.5, 1.0), (None)),
        ((0.25, 0.5), (0.25, 1.0)),
        ((0.5, 0.75), (0.25, 1.0)),
        ((0.75, 1.0), (0.25, 1.0)),
    ]

    fig = plt.figure(figsize=(8, 10))
    subplots = [322, 323, 324, 325]
    for i, ratio in enumerate(ratios):
        plt.subplot(subplots[i])
        ax = plt.gca()
        plot_grayscale_ratio_rosette(*ratio, ax=ax)

    plt.subplot(321)
    plt.subplot(322)

    plt.tight_layout()
    plt.show()

# %%
from copy import deepcopy
if __name__ == "__main__":
    rescaled_model = model.rescale(0.6)
    regridded_rescaled_model = deepcopy(rescaled_model)
    regridded_rescaled_model.regrid()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))
    model.plot_intensity(0, ax=ax0, axis_length=400)
    rescaled_model.plot_intensity(0, ax=ax1, axis_length=400)
    regridded_rescaled_model.plot_intensity(0, ax=ax2, axis_length=400)

    fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15, 5))
    z_unsc, amp_unsc = list(model.amplitudes.items())[0]
    z_sc, amp_sc = list(rescaled_model.amplitudes.items())[0]
    z_sc_rg, amp_sc_rg = list(regridded_rescaled_model.amplitudes.items())[0]
    amp_unsc.intensity.plot(ax=ax3, axis_length=400)
    ax3.set_title(f"Unscaled {z_unsc*1e3:.2f} mm")
    amp_sc.intensity.plot(ax=ax4, axis_length=400)
    ax4.set_title(f"Scaled {z_sc*1e3:.2f} mm")
    amp_sc_rg.intensity.plot(ax=ax5, axis_length=400)
    ax5.set_title(f"Scaled and regridded {z_sc_rg*1e3:.2f} mm")

# %%
