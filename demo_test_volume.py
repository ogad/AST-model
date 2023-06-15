# %%
import numpy as np
import logging
from tqdm import tqdm

from psd_ast_model import GammaPSD
from volume_model import CloudVolume

logging.basicConfig(level=logging.INFO)


psd = GammaPSD(1e10, 2e2, 2.5, bins=np.logspace(-8, -3, 10000))
cloud = CloudVolume(psd, (1, 1, 1))

print(cloud.n_particles)
# %%
import matplotlib.pyplot as plt
# slice = cloud.slice(200)
# plt.imshow(slice.intensity[4000:8000, 4000:8000])

pcle = cloud.particles.iloc[0]
detector_location = pcle.position - np.array([0, 1.1* pcle.diameter, 5e-2])

images = [cloud.take_image(detector_location + np.array([0, y, 0])) for y in tqdm(np.arange(0, 2.2*pcle.diameter, 10e-6))]
interesting_images = [img for img in images if (img.intensity != 1).any()]
image = np.concatenate([img.intensity for img in images], axis=1)
plt.imshow(image)
# %%
