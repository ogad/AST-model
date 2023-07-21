# Data generation for compression project
# Author: Oliver Driver
# Date: 05/07/2023

# %% 
# Imports
import random
import logging

from ast_model import plot_outline
logging.basicConfig(level=logging.INFO)

import numpy as np
import matplotlib.pyplot as plt

from psd_ast_model import GammaPSD, CompositePSD, CrystalModel
from cloud_model import CloudVolume, Detector
from detector_model import Detector

from tqdm import tqdm

np.random.seed(42)
random.seed(42)

# %%
# Gamma distributions
gamma_dist_base = GammaPSD.from_concentration(8000, 8.31e4, 7.86)
gamma_dist = GammaPSD.from_concentration(8, 8.31e4, 7.86)

gammas = {
    "Spheres": GammaPSD.from_mean_variance(500e6, 25e-6, 10e-6**2),
    "Columns": GammaPSD.from_mean_variance(200e3, 200e-6, 100e-6**2, model=CrystalModel.RECT_AR5),
    "Flakes": GammaPSD.from_mean_variance(1e3, 2000e-6, 300e-6**2, model=CrystalModel.ROS_6),
}
for gamma in gammas.values():
    gamma.bins = gammas["Flakes"].bins
composite_psd = CompositePSD(list(gammas.values()), bins=gammas["Flakes"].bins)

ax = plt.axes()
for shape, gamma in gammas.items():
    gamma.plot(ax=ax, label=f"{shape}; {gamma.total_number_density * 1e-3:.0e}/L")
composite_psd.plot(ax=ax,label="Composite")
plt.xscale("log")
plt.yscale("log")
plt.ylim(1e-4, 1e13)
plt.legend()
plt.tight_layout()
plt.show()

# %%

cloud = CloudVolume(composite_psd, (0.1,1,0.2))

# %%
# import pickle
# with open(f"../data/composite_cloud.pkl", "wb") as f:
#         pickle.dump(cloud, f)
# %%
runs = []
arms = [0.06, 0.2]

for arm_sep in arms:
    logging.info(f"Arm separation: {arm_sep}...")
    detector = Detector(np.array([0.05, 0.5, 0.1-arm_sep/2]), n_pixels=256, pixel_size=10e-6, arm_separation=arm_sep)
    run = cloud.take_image(detector, distance=0.01, single_image=True)
    runs.append(run)
# run.save(f"../data/{datetime.datetime.now():%Y-%m-%d}_{run.distance}_composite_run.pkl")
# %%

fig, axs = plt.subplots(1, len(runs), figsize=(10,20))
for i, run in enumerate(runs):
    plt.axes(axs[i])
    run.images[0].plot(ax=axs[i],grayscale_bounds=[0.35,0.5,0.65], colorbar=False)
plt.show()
# %%
8* 25e-6**2/(4*detector.wavelength)
# %%
