# %%
import logging
from random import seed
import pickle

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from ast_model import plot_outline
from psd_ast_model import GammaPSD, TwoMomentGammaPSD
from volume_model import CloudVolume, Detector

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
except FileNotFoundError:
    cloud = CloudVolume(gamma_dist, (0.1, 1000, 0.1))
    with open("cloud_01_1000_01.pkl", "wb") as f:
        pickle.dump(cloud, f)

print(cloud.n_particles)
# slice = cloud.slice(200)
# plt.imshow(slice.intensity[4000:8000, 4000:8000])
# %%
pcle = cloud.particles.iloc[0]
detector_location = pcle.position - np.array([300e-6, 290*pcle.diameter, 4e-2])

# images = [cloud.take_image(detector_location + np.array([0, y, 0])) for y in tqdm(np.arange(0, 2.2*pcle.diameter, 10e-6))]
# interesting_images = [img for img in images if (img.intensity != 1).any()]
# image = np.concatenate([img.intensity for img in images], axis=1)

n_pixels = 128

detector_1 = Detector(detector_location, n_pixels=n_pixels)

image = cloud.take_image(detector_1, distance=300* pcle.diameter).amplitude.intensity
plt.imshow(image)
plt.scatter(0, n_pixels / 2, c="r")
plt.colorbar()


# %%

detector = Detector(np.array([0.05, 0.1, 0]))
detections, particles = cloud.take_image(detector, distance=10, separate_particles=True)
objects, _ = cloud.take_image(detector, distance=10, separate_particles=True, use_focus=True)

# detections.amplitude.intensity.plot()
# %%
# colormap from red to transparent
from matplotlib.colors import LinearSegmentedColormap
object_cmap = LinearSegmentedColormap.from_list("red_transparent", [(1, 0, 0, 1), (1, 0, 0, 0)])
object_norm = plt.Normalize(0, 1)


diameters = []
for image, object in zip(detections, objects):
    if image.amplitude.intensity.min() <= 0.5:
        image.plot(grayscale_bounds=[0.5])
        ax = plt.gca()

        measured_diameter = image.measure_diameters()
        accurate_diameter = object.measure_diameters()
        z = (image.particles.iloc[0].position[2] - detector.position[2] - detector.arm_separation/2) * 1e2

        plt.text(20, 20,
            f"z = {z:.1f} cm\nNo. regions = {len(measured_diameter)}\
            \nMeasured diameter = { ','.join(f'{s:.0f}' for s in list(measured_diameter.values())) } µm\
            \nObject diameter = {','.join(f'{s:.0f}' for s in list(accurate_diameter.values()))} µm",
            ha="left", va="bottom", bbox=dict(facecolor='white', alpha=0.5), 
        )
        plt.xlim(0, 1280)
        plot_outline(object.amplitude.intensity.T<0.1, ax)
        plt.tight_layout()
        plt.show()
        diameters.append(measured_diameter)
