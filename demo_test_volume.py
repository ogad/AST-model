# %%
import logging
from random import seed
import pickle

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime

from ast_model import plot_outline
from psd_ast_model import GammaPSD, TwoMomentGammaPSD, CrystalModel
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
@profile(f"../data/profile__take_image__{datetime.datetime.now():%Y-%m-%d_%H%M}.prof")
def take_image(detector, distance, cloud: CloudVolume, separate_particles):
    return cloud.take_image(detector, distance=distance, separate_particles=separate_particles)

detector_run_version=1
redo_detections = True
shape = CrystalModel.RECT_AR5_ROT
distance = 999
n_px = 128

try:
    run = DetectorRun.load(f"../data/run_v{detector_run_version}_{distance}_{n_px}px_{shape.name}_run.pkl")
except FileNotFoundError:
    detector = Detector(np.array([0.05, 0.1, 0]), n_pixels=n_px)
    cloud.particles.loc[:, "model"] = shape
    # run = cloud.take_image(detector, distance=distance, separate_particles=True)
    run = take_image(detector, distance, cloud, True)
    run.save(f"../data/run_v{detector_run_version}_{distance}_{n_px}px_{shape.name}_run.pkl")

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
retrieval2 = Retrieval(run, DiameterSpec(diameter_method="xy", min_sep=5e-4, filled=True))
retrieval = Retrieval(run, DiameterSpec(min_sep=5e-4))

# retrieval = Retrieval(run, DiameterSpec(diameter_method="xy", min_sep=0.1, filled=True))
fit = retrieval.fit_gamma(min_diameter = 20e-6) # What minimum diameter is appropriate; how can we account for the low spike...
fit2 = retrieval2.fit_gamma(min_diameter = 20e-6)

# %%
fig, axs = plt.subplots(2, 2, width_ratios=[1, 5], height_ratios=[3,1], figsize=(10, 7), sharex='col')

psd_axs = axs[0,:]

for ax in psd_axs:
    true = gamma_dist.plot(ax, label=f"True\n{gamma_dist.parameter_description()}")
    retrieval.plot(label="Retrieved (Circ. equiv.)", ax=ax, color="C1")
    retrieval2.plot(label="Retrieved (XY)", ax=ax, color="C2")
    fit_ce = fit.plot(ax, label=f"Circle equivalent\n{fit.parameter_description()}", color="C1")
    fit_xy = fit2.plot(ax, label=f"XY mean\n{fit2.parameter_description()}", color="C2")

# plt.yscale("log")
psd_axs[0].set_xlim(0, 1e-4)
psd_axs[0].set_xticks([0, 1e-4])
psd_axs[1].set_ylim(0, 0.5e9)
psd_axs[1].set_xlim(0, 5e-4)
psd_axs[1].legend(handles=true+fit_ce+fit_xy)
psd_axs[0].get_legend().remove()

axs[1][1].bar(retrieval.midpoints, np.histogram(retrieval.diameters, bins=retrieval.bins)[0], width=0.9*np.diff(retrieval.bins), color="C1", alpha=0.2)
axs[1][1].bar(retrieval2.midpoints, np.histogram(retrieval2.diameters, bins=retrieval.bins)[0], width=0.9*np.diff(retrieval.bins), color="C2", alpha=0.2)
axs[1][1].set_xlabel("Diameter (m)")
axs[1][1].set_ylabel("Count")

axs[1,0].remove()
# plt.show()

# %%
