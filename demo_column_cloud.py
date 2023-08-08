# %%
import logging
from random import seed
import pickle

from tqdm.autonotebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

from ast_model import plot_outline, ASTModel
from psd_ast_model import GammaPSD, TwoMomentGammaPSD, CrystalModel
from cloud_model import CloudVolume, Detector

logging.basicConfig(level=logging.INFO)

# %% Retry with a more realistic gamma distribution
# reinitialise the random seed
seed(42)
np.random.seed(42)

gamma_dist = GammaPSD(1.17e43, 8.31e4, 7.86)

# %%
# psd.plot(ax)
try:
    with open("cloud_01_1000_01.pkl", "rb") as f:
        cloud = pickle.load(f)
except FileNotFoundError:
    cloud = CloudVolume(gamma_dist, (0.1, 1000, 0.1))
    with open("cloud_01_1000_01.pkl", "wb") as f:
        pickle.dump(cloud, f)

cloud.particles.model[:] = CrystalModel.RECT_AR5
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

for image, object in zip(detections, objects):
    if image.amplitude.intensity.min() < 0.5:
        image.plot(grayscale_bounds=[0.5])
        ax = plt.gca()
        # object.plot(grayscale_bounds=[0.5], cmap=object_cmap, ax=ax, alpha=0.3, norm=object_norm, colorbar=False)

        measured_diameter = image.measure_diameters()
        # logging.info(f"Measured diameter: {measured_diameter:.2f} µm; Actual diameter: {image.particles.iloc[0].diameter*1e6:.2f} µm")
        
        # plt.errorbar(image.particles.x_index*10, image.particles.y_index * 10, xerr=image.particles.diameter/2e-6, yerr=image.particles.diameter/2e-6,capsize=5, fmt="o", c="r")

        # abs_y = lambda rel_y: (image.particles.iloc[0].position[1] + (5*image.particles.iloc[0].diameter - rel_y*1e-6))
        # rel_y = lambda abs_y: (5*image.particles.iloc[0].diameter - (abs_y - image.particles.iloc[0].position[1]))/1e-6
        # ax = plt.gca()
        # secax = ax.secondary_yaxis('right', functions=(abs_y, rel_y))
        # secax.set_ylabel("y (m)")

        z = (image.particles.iloc[0].position[2] - detector.position[2] - detector.arm_separation/2) * 1e2 # FIXME: this wont work for loaded clouds because image.particles doesn't exist.
        plt.text(20, 20, f"z = {z:.1f} cm\nNo. regions = {len(measured_diameter)}", ha="left", va="bottom", bbox=dict(facecolor='white', alpha=0.5), )
        plt.xlim(0, 1280)
        plot_outline(object.amplitude.intensity.T<0.1, ax)
        plt.tight_layout()
        plt.show()
        logging.info(f"next...")

# %%
particles["x"] = particles.position.apply(lambda pos: pos[0])
particles["y"] = particles.position.apply(lambda pos: pos[1])
particles["z"] = particles.position.apply(lambda pos: pos[2])
# %%
particles.head()
# %%
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection="3d")
ax.scatter(particles.x, particles.y, particles.z, c=particles.diameter, cmap="viridis")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")

# %%
particles
# %%
