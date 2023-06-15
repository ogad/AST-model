# %%
import numpy as np
import logging

from psd_ast_model import GammaPSD
from volume_model import CloudVolume

logging.basicConfig(level=logging.INFO)


psd = GammaPSD(1e10, 2e2, 2.5, bins=np.logspace(-8, -3, 10000))
cloud = CloudVolume(psd, (0.05, 0.05, 500))

print(cloud.n_particles)
# %%
import matplotlib.pyplot as plt
slice = cloud.slice(200)
plt.imshow(slice.intensity[4000:8000, 4000:8000])
# %%
