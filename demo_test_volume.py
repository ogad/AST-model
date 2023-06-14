# %%
import numpy as np
import logging

from psd_ast_model import GammaPSD
from volume_model import CloudVolume

logging.basicConfig(level=logging.INFO)


psd = GammaPSD(1e10, 2e2, 2.5, bins=np.logspace(-8, -3, 10000))
cloud = CloudVolume(psd, (0.1, 0.05, 2e3))

print(cloud.n_particles)
# %%
