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
from cloud_model import CloudVolume, Detector
from detector_model import Detector, ImagedRegion, ImageFilter, DiameterSpec
from retrieval_model import Retrieval
from detector_run import DetectorRun

from profiler import profile

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
cloud_len = 1000
try:
    with open(f"cloud_01_{cloud_len}_01.pkl", "rb") as f:
        cloud = pickle.load(f)
except (FileNotFoundError, ModuleNotFoundError):
    cloud = CloudVolume(gamma_dist, (0.1, cloud_len, 0.1))
    with open(f"cloud_01_{cloud_len}_01.pkl", "wb") as f:
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
from psd_ast_model import CrystalModel
redo_detections = True
shape = "rects"
distance = 999
n_px = 128
if redo_detections:
    detector = Detector(np.array([0.05, 0.1, 0]), n_pixels=n_px)
    cloud.particles.loc[:, "model"] = CrystalModel.RECT_AR5 if shape == "rects" else CrystalModel.SPHERE
    run = cloud.take_image(detector, distance=distance, separate_particles=True)
    run.save(f"../data/{datetime.datetime.now():%Y-%m-%d}_{run.distance}_{n_px}px_{shape}_run.pkl")
else:
    run = DetectorRun.load("../data/2023-07-11_999_spheres_run.pkl")
    # run = DetectorRun.load("../data/2023-07-06_999_rects_run.pkl")
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
# diameter_series["50%"] = run.measure_diameters(spec=DiameterSpec(diameter_method="xy", edge_filter=False, framed=False, bound=False))
# diameter_series["50% framed"] = run.measure_diameters(spec=DiameterSpec(diameter_method="xy", edge_filter=False, bound=False))
# diameter_series["50% no edge"] = run.measure_diameters(spec=DiameterSpec(diameter_method="xy",  bound=False, framed=False))
# diameter_series["50% no edge bounded"] = run.measure_diameters(spec=DiameterSpec(diameter_method="xy", filled=True, framed=False))
# diameter_series["50% no edge bounded unfilled"] = run.measure_diameters(spec=DiameterSpec(diameter_method="xy", framed=False))
# diameter_series["50% no edge bounded framed"] = run.measure_diameters(spec=DiameterSpec(diameter_method="xy", filled=True))
diameter_series["50% no edge bounded framed minsep"] = run.measure_diameters(spec=DiameterSpec(diameter_method="xy", min_sep=0.001, filled=True))
# diameter_series["50% no edge, circle equiv. bounded unfilled framed"] = run.measure_diameters(spec=DiameterSpec(min_sep=0.001))

bins = np.linspace(0, 5e-4, 50)

# gamma_dist.plot(plt.gca(), label="True")

for label, diameters in diameter_series.items():
    diameters = list(diameters.values())
    plt.stairs(np.histogram(np.array(diameters) * 1e-6, bins=bins)[0] / (np.diff(bins) * run.volume(bins[1:])), bins, label=label)
# plt.legend()
# plt.xlim(0,5e-4)
# # plt.yscale("log")

# plt.xlabel("Diameter (m)")
# plt.ylabel("dN/dD ($\mathrm{m}^{-3}\,\mathrm{m}^{-1}$)")#, color="C0")
# plt.ylim(0, 0.5e9)
# plt.yticks(color="C0")

# plt.show()
# %%
retrieval2 = Retrieval(run, DiameterSpec(diameter_method="xy", min_sep=0.1, filled=True))
retrieval = Retrieval(run, DiameterSpec(min_sep=0.1))

# retrieval = Retrieval(run, DiameterSpec(diameter_method="xy", min_sep=0.1, filled=True))
fit = retrieval.fit_gamma(min_diameter = 20e-6) # What minimum diameter is appropriate; how can we account for the low spike...
fit2 = retrieval2.fit_gamma(min_diameter = 20e-6)

# %%
fig, axs = plt.subplots(1, 2, width_ratios=[1, 5], figsize=(10, 5))

for ax in axs:
    retrieval.plot(label="Retrieved (Circ. equiv.)", ax=ax)
    retrieval2.plot(label="Retrieved (XY)", ax=ax)
    gamma_dist.plot(ax, label=f"True\n{gamma_dist.parameter_description()}")
    fit.plot(ax, label=f"Fit (Circ. equiv.)\n{fit.parameter_description()}")
    fit2.plot(ax, label=f"Fit (XY)\n{fit2.parameter_description()}")

# plt.yscale("log")
axs[0].set_xlim(0, 1e-4)
axs[0].set_xticks([0, 1e-4])
axs[1].set_ylim(0, 0.5e9)
axs[1].set_xlim(0, 5e-4)
axs[1].legend()
axs[0].get_legend().remove()
plt.show()

# %%
def continuous_int_input():
    while True:
        try:
            inp = input("Enter a number, or 'q' to quit:")
            if inp == "q":
                break
            yield int(inp)
        except ValueError:
            print("Please enter a number")

# %%
# coord = list(retrieval.detected_particles.keys())[7]
# cloud.plot_from_run(run, 1e-6*np.array(coord), grayscale_bounds=[.5])

# locations = [coord for coord, diameter in retrieval.detected_particles.items() if diameter < 50]

# fig, axs = plt.subplots(len(locations), 2, figsize=(7, 4*len(locations)))
# for i, location in enumerate(locations):
#     cloud.plot_from_run(run, 1e-6*np.array(location), ax=axs[i][0], colorbar=True)
#     cloud.plot_from_run(run, 1e-6*np.array(location), ax=axs[i][1], grayscale_bounds=[.5], plot_outlines=True, cloud=cloud)
# plt.tight_layout()
# plt.show()

# locations_to_remove = [locations[i] for i in continuous_int_input()]
# retrieval.remove_particles(locations_to_remove)


# %%
# Plan of attack:
# Bucket our diameters into bins (which bins?)
# Work out a SVol for each bin (EAW approx constant with z)
# Work out a number density for each bin, and then divide by the bin width to give an instantanous dN/dD
# %%
# sebs papers size metric z invariant ish. circ equivalent.
# %%
