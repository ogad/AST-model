# %%
import logging
from random import seed
import pickle

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime

from ast_model import plot_outline
from psd_ast_model import GammaPSD, TwoMomentGammaPSD
from cloud_model import CloudVolume, Detector, DetectorRun

logging.basicConfig(level=logging.INFO)

# %% Retry with a more realistic gamma distribution
# reinitialise the random seed
seed(42)
np.random.seed(42)

# O'Shea 2021 Fig. 18(c)
# N_0 = 10e4 L-1 cm-1; µ = 2; @lambda = 200 cm-1
# gamma_dist = OSheaGammaPSD(1e4 * 1e3 * 1e2, 200 * 1e2, 2)

# gamma_dist = OSheaGammaPSD(102 * 1e3 * 1e2, 4.82 * 1e2, 2.07, bins=np.logspace(-8, -2, 10000))
# gamma_dist = TwoMomentGammaPSD.from_m2_tc(8e-4, -40, bins=np.logspace(-8, -1, 10000))
gamma_dist = GammaPSD(1.17e43, 8.31e4, 7.86)

fig, ax = plt.subplots()
gamma_dist.plot(ax)
# %%
# psd.plot(ax)
try:
    with open("cloud_01_1000_01.pkl", "rb") as f:
        cloud = pickle.load(f)
except (FileNotFoundError, ModuleNotFoundError):
    cloud = CloudVolume(gamma_dist, (0.1, 1000, 0.1))
    with open("cloud_01_1000_01.pkl", "wb") as f:
        pickle.dump(cloud, f)

print(cloud.n_particles)
# %%
pcle = cloud.particles.iloc[0]
detector_location = pcle.position - np.array([300e-6, 15*pcle.diameter, 4e-2])

n_pixels = 128

detector_1 = Detector(detector_location, n_pixels=n_pixels)

image = cloud.take_image(detector_1, distance=30* pcle.diameter).amplitude.intensity.field
plt.imshow(image)
plt.scatter(0, n_pixels / 2, c="r")
plt.colorbar()


# %%
redo_detections = False
if redo_detections:
    detector = Detector(np.array([0.05, 0.1, 0]))
    run = cloud.take_image(detector, distance=990, separate_particles=True)
else:
    run = DetectorRun.load("../data/2023-06-29_10_spheres_run.pkl")
# objects, _ = cloud.take_image(detector, distance=10, separate_particles=True, use_focus=True)

# detections.amplitude.intensity.plot()
# %%
# colormap from red to transparent
from matplotlib.colors import LinearSegmentedColormap
object_cmap = LinearSegmentedColormap.from_list("red_transparent", [(1, 0, 0, 1), (1, 0, 0, 0)])
object_norm = plt.Normalize(0, 1)


run.plot(grayscale_bounds=[0.5], plot_outlines=True,  cloud=cloud)

# %%
diameters = run.measure_diameters()
bins = np.linspace(1e-5, 1e-3, 40)
plt.plot(bins[:-1], np.histogram(np.array(diameters) * 1e-6, bins=bins)[0] / (np.diff(bins)), color="C1", label="Measured")
plt.ylabel("Measured particles/bin width", color="C1")
plt.yticks(color="C1")

ax2 = plt.gca().twinx()
gamma_dist.plot(plt.gca())
# plt.xscale("log")

plt.xlabel("Diameter (m)")
plt.ylabel("dN/dD (m$^{-1}$)", color="C0")
plt.yticks(color="C0")

# plt.show()

#  # %% Saving the detections
# run.save(f"../data/{datetime.datetime.today():%Y-%m-%d}_{run.distance}_spheres_run.pkl")


# # %% Load the detections for postprocessing
# run_loaded = DetectorRun.load("../data/2023-06-28_detections.pkl")
# %%
# sample_length = 10 # m
# effective_array_width = # ? m: pixel_size * (n_pixels - 1) - diameter (parallel to array?)
# depth_of_field = # ? m ± cD^2/4λ; c = 8 ish for 2D-S. (from Gurganus Lawson 2018)
# sample_volume = sample_length * effective_array_width * depth_of_field # should strictly be integrated...

# Plan of attack:
# Bucket our diameters into bins (which bins?)
# Work out a SVol for each bin (EAW approx constant with z)
# Work out a number density for each bin, and then divide by the bin width to give an instantanous dN/dD
# %%
# sebs papers size metric z invariant ish. circ equivalent.
# %%
