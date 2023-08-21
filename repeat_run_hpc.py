
#%%
import os
import numpy as np
import logging
import sys

from oap_model.psd import CrystalModel, GammaPSD
from oap_model.cloud import CloudVolume
from oap_model.detector import Detector

gammas = {
    ("in-situ", "cold"): GammaPSD.w19_parameterisation(-70, total_number_density=44, insitu_origin=True),
    ("in-situ", "hot"): GammaPSD.w19_parameterisation(-50, total_number_density=44, insitu_origin=True),
    ("liquid", "cold"): GammaPSD.w19_parameterisation(-60, total_number_density=44, liquid_origin=True),
    ("liquid", "hot"): GammaPSD.w19_parameterisation(-40, total_number_density=44, liquid_origin=True),
}

gamma_labels = [
    ("in-situ", "cold"),
    ("in-situ", "hot"),
    ("liquid", "cold"),
    ("liquid", "hot"),
]

def produce_repeat(i, gamma_spec=gammas.keys(), distance=10_000, n_px=128):
    for psd in gamma_spec:
        for shape in [CrystalModel.SPHERE, CrystalModel.RECT_AR5]:
            logging.info(f"Running {psd} {shape.name}, repeat {i}")
            file_name = f"run_{distance}_{'-'.join(psd)}_{n_px}px_{shape.name}_repeat{i}".replace(".", "_")
            cloud = CloudVolume(gammas[psd], (0.1, distance+1, 0.21), random_seed=i)
            cloud.set_model(shape)
            detector = Detector(np.array([0.05, 0.1, 0.01]), n_pixels=128, arm_separation=0.06, pixel_size=10e-6, detection_length=128*10e-6)
            run = cloud.take_image(detector, distance=distance, single_image = False, binary_output=True)
            
            outputfile = os.path.join(os.path.dirname(__file__), f"data/{file_name}.pkl")
            run.save(outputfile)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    i =int(os.environ["PBS_ARRAY_INDEX"])
    i_gamma = int(i) % len(gamma_labels)
    i_repeat = int(i) // len(gamma_labels)
    produce_repeat(i_repeat, [gamma_labels[i_gamma]], distance=10_000)
# %%
