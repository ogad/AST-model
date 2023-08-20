# %%
import os

from detector_model import DiameterSpec
from detector_run import DetectorRun
from retrieval_model import Retrieval


path = "../data_hpc/"

for file in os.listdir(path):
    try:
        run = DetectorRun.load(path + file)
        print(len(run.images))
        run.images[5].plot()
        retrieval = Retrieval(run, DiameterSpec(min_sep=5e-4))
        print(list(retrieval.detected_particles.items())[:5])
    except:
        print("Error with file: " + file)
        continue
# %%
