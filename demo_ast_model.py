# %%

# Package imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Local imports
from ast_model import ASTModel


# define the object
image = np.zeros((16, 16))
image[7:9, 7:9] = 1

# create the model
model = ASTModel(opaque_shape=object)

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
ax1.set_xlabel("x")
ax1.set_ylabel("y")

## Plot the diffracted intensity on second axis
ax2.imshow(intensity, norm=intensity_norm, cmap="BrBG_r")
ax2.set_title(f"Intensity at z = {z*1000}mm")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
plt.tight_layout()

plt.show()
