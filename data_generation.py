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

gamma_test  = GammaPSD.from_mean_variance(8000, 20e-6, 0.000002**2)
gamma_test.plot(plt.axes())
plt.show()
# %%
ax = plt.axes()
gamma_dist_base.plot(ax=ax)
gamma_dist.plot(ax=ax)
plt.yscale("log")
# plt.xscale("log")
# %%
