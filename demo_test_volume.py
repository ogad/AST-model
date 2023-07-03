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
from detector_model import Detector, ImagedRegion, DetectorRun, ImageFilter

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
from cloud_model import CrystalModel
redo_detections = False
if redo_detections:
    detector = Detector(np.array([0.05, 0.1, 0]))
    cloud.particles.loc[:, "model"] = CrystalModel.RECT_AR5
    run = cloud.take_image(detector, distance=5, separate_particles=True)
    run.save(f"../data/{datetime.datetime.now():%Y-%m-%d}_{run.distance}_rects_run.pkl")
else:
    run = DetectorRun.load("../data/2023-06-29_999_spheres_run.pkl")
# objects, _ = cloud.take_image(detector, distance=10, separate_particles=True, use_focus=True)

# detections.amplitude.intensity.plot()
# %%
# colormap from red to transparent
from matplotlib.colors import LinearSegmentedColormap
object_cmap = LinearSegmentedColormap.from_list("red_transparent", [(1, 0, 0, 1), (1, 0, 0, 0)])
object_norm = plt.Normalize(0, 1)

if run.distance <= 100:
    run.plot(grayscale_bounds=[0.25, 0.5, 0.75], plot_outlines=True,  cloud=cloud)
    plt.tight_layout()

# %%

diameter_series = {}
diameter_series["50%"] = run.measure_diameters(image_filters=[ImageFilter.PRESENT_HALF_INTENSITY])
# diameter_series["50% framed"] = run.measure_diameters(image_filters=[ImageFilter.PRESENT_HALF_INTENSITY], type="xy_framed")#FIXME: why is this different to the unframed case?
diameter_series["50% no edge"] = run.measure_diameters(image_filters=[ImageFilter.PRESENT_HALF_INTENSITY, ImageFilter.NO_EDGE_HALF_INTENSITY])
diameter_series["50% no edge bounded"] = run.measure_diameters(image_filters=[ImageFilter.PRESENT_HALF_INTENSITY, ImageFilter.NO_EDGE_HALF_INTENSITY], bounded=True)
diameter_series["50% no edge bounded framed"] = run.measure_diameters(image_filters=[ImageFilter.PRESENT_HALF_INTENSITY, ImageFilter.NO_EDGE_HALF_INTENSITY], bounded=True, type="xy_framed")

bins = np.linspace(0, 5e-4, 50)

gamma_dist.plot(plt.gca(), label="True")

for label, diameters in diameter_series.items():
    plt.stairs(np.histogram(np.array(diameters) * 1e-6, bins=bins)[0] / (np.diff(bins) * run.volume(bins[1:])), bins, label=label)
plt.legend()
plt.xlim(0,5e-4)
# plt.yscale("log")sssß

plt.xlabel("Diameter (m)")
plt.ylabel("dN/dD ($\mathrm{m}^{-3}\,\mathrm{m}^{-1}$)")#, color="C0")
plt.ylim(0, 0.5e9)
# plt.yticks(color="C0")

# plt.show()
# %%
# Plan of attack:
# Bucket our diameters into bins (which bins?)
# Work out a SVol for each bin (EAW approx constant with z)
# Work out a number density for each bin, and then divide by the bin width to give an instantanous dN/dD
# %%
# sebs papers size metric z invariant ish. circ equivalent.
# %%
