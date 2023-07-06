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

gammas = {
    "Spheres": GammaPSD.from_mean_variance(500e6, 25e-6, 25e-6**2),
    "Columns": GammaPSD.from_mean_variance(200e3, 200e-6, 100e-6**2),
    "Flakes": GammaPSD.from_mean_variance(1e3, 3000e-6, 1000e-6**2),
}
for gamma in gammas.values():
    gamma.bins = gammas["Flakes"].bins
ax = plt.axes()
for shape, gamma in gammas.items():
    gamma.plot(ax=ax, label=f"{shape}; {gamma.total_number_density * 1e-3:.0e}/L")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-4, 1e13)
plt.legend()
plt.tight_layout()
plt.show()
# %%
ax = plt.axes()
gamma_dist_base.plot(ax=ax)
gamma_dist.plot(ax=ax)
plt.yscale("log")
# plt.xscale("log")
# %%
