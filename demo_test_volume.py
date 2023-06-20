# %%
import logging
from random import seed

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from psd_ast_model import GammaPSD
from volume_model import CloudVolume

logging.basicConfig(level=logging.INFO)

# %% Old demo with unrealistically high density
seed(42)
np.random.seed(42)


psd = GammaPSD(1e10, 2e2, 0.5, bins=np.logspace(-8, -3, 10000))
cloud = CloudVolume(psd, (0.1, 0.1, 0.2))

print(cloud.n_particles)
# slice = cloud.slice(200)
# plt.imshow(slice.intensity[4000:8000, 4000:8000])

pcle = cloud.particles.iloc[0]
detector_location = pcle.position - np.array([-800e-6, - 0.8* pcle.diameter, 4e-2])

# images = [cloud.take_image(detector_location + np.array([0, y, 0])) for y in tqdm(np.arange(0, 2.2*pcle.diameter, 10e-6))]
# interesting_images = [img for img in images if (img.intensity != 1).any()]
# image = np.concatenate([img.intensity for img in images], axis=1)
image = cloud.take_image(detector_location, distance=2.2*pcle.diameter).amplitude.intensity
plt.imshow(image)
plt.colorbar()

# %% Retry with a more realistic gamma distribution
# reinitialise the random seed
seed(42)
np.random.seed(42)

# O'Shea 2021 Fig. 18(c)
# N_0 = 10e4 L-1 cm-1; µ = 2; @lambda = 200 cm-1
gamma_dist = GammaPSD(1e4 * 1e3 * 1e2, 200 * 1e2, 2)

fig, ax = plt.subplots()
gamma_dist.plot(ax)
# psd.plot(ax)
cloud = CloudVolume(gamma_dist, (0.1, 0.1, 1000))

print(cloud.n_particles)
# slice = cloud.slice(200)
# plt.imshow(slice.intensity[4000:8000, 4000:8000])
# %%
pcle = cloud.particles.iloc[0]
detector_location = np.array([0.05, 0.05, 10])

for z in range(10, 1000, 10):
    detection = cloud.take_image(detector_location, distance=10)
    if detection:
        intensity = image.amplitude.intensity
        plt.imshow(image)
        plt.colorbar()
        plt.show()
        detector_location[2] = z + 10

# plt.imshow(image)
# plt.colorbar()



# %%
pcle.diameter
# gamma_dist.total_number_density
# %%
# Use the O'Shea formulation exactly to check the units are working...
n_gamma_oshea_units = lambda N, µ, l, d: N * d**µ * np.exp(-l*d)
diameters = np.linspace(0, 1e-1, 1000)
n_gamma_vals = n_gamma_oshea_units(1e4, 2, 200, diameters)

plt.plot(diameters * 1e4, n_gamma_vals)
plt.xscale('log')
plt.yscale('log')

# %%
# per litre
np.sum(n_gamma_vals[1:] * np.diff(diameters)) * 1e3
# %%
