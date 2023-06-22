# %%
import logging
from random import seed
import pickle

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
detections = cloud.take_image(detector, distance=10, separate_particles=True)

# detections.amplitude.intensity.plot()

# %%
for image in detections:
    if image.amplitude.intensity.min() < 0.5:
        # image.amplitude.intensity.plot(grayscale_bounds=[0.5])

        measured_diameter = image.amplitude.intensity.measure_xy_diameter()
        logging.info(f"Measured diameter: {measured_diameter:.2f} µm")
        
        # plt.errorbar(image.particles.x_index*10, image.particles.y_index * 10, xerr=image.particles.diameter/2e-6, yerr=image.particles.diameter/2e-6,capsize=5, fmt="o", c="r")

        # abs_y = lambda rel_y: (image.particles.iloc[0].position[1] + (5*image.particles.iloc[0].diameter - rel_y*1e-6))
        # rel_y = lambda abs_y: (5*image.particles.iloc[0].diameter - (abs_y - image.particles.iloc[0].position[1]))/1e-6
        # ax = plt.gca()
        # secax = ax.secondary_yaxis('right', functions=(abs_y, rel_y))
        # secax.set_ylabel("y (m)")

        # plt.xlim(0, 1280)

        # plt.show()

# %%
