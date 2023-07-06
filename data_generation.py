# Data generation for compression project
# Author: Oliver Driver
# Date: 05/07/2023

# %% 
# Imports
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import matplotlib.pyplot as plt

from ast_model import plot_outline
from psd_ast_model import GammaPSD, TwoMomentGammaPSD
from cloud_model import CloudVolume, Detector, DetectorRun
from detector_model import Detector, ImagedRegion, DetectorRun, ImageFilter, DiameterSpec

# %%
# Gamma distributions
gamma_dist_base = GammaPSD.from_concentration(8000, 8.31e4, 7.86)
gamma_dist = GammaPSD.from_concentration(8, 8.31e4, 7.86)

gammas = [
    GammaPSD.from_mean_variance(20e6, 20e-6, 10e-6**2),
    GammaPSD.from_mean_variance(100, 300e-6, 100e-6**2),
]
ax = plt.axes()
for gamma in gammas:
    gamma.plot(ax=ax)
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()
# %%
ax = plt.axes()
gamma_dist_base.plot(ax=ax)
gamma_dist.plot(ax=ax)
plt.yscale("log")
# plt.xscale("log")
# %%
