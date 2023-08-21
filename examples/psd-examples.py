# %%
import itertools
from os import access
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

from oap_model.detector_run import DetectorRun
from oap_model.psd import CrystalModel
from oap_model.detector import DiameterSpec, Detector
from oap_model.retrieval import Retrieval

from examples.repeat_run_hpc import gammas
from residuals_analysis import detector_kwargs, diameter_spec_kwargs

base_run_len = 10000


if __name__ == '__main__':
    fig, axs = plt.subplots(1,2, figsize=(7.2, 2.3))
    i_color=0

    for id_tuple in tqdm(itertools.product([('in-situ','cold'),('liquid','cold')], [CrystalModel.RECT_AR5], ["2d128"])):
        
        gammas_labels, shape, instrument = id_tuple

        detector = Detector(np.array([0.05, 0.1, 0.01]), **detector_kwargs[instrument])

        gamma = gammas[gammas_labels]

        shape = CrystalModel.SPHERE
        for shape, ax in zip([CrystalModel.SPHERE, CrystalModel.RECT_AR5], axs):
            # if ('in-situ','cold') in id_tuple and shape==CrystalModel.SPHERE:
            #     continue

            base_run = DetectorRun.load(f"../data/{'-'.join(gammas_labels)}/run_{base_run_len}_{'-'.join(gammas_labels)}_128px_{shape.name}_repeat0.pkl")
            base_run.detector = detector
            retrievals = Retrieval(base_run, DiameterSpec(**diameter_spec_kwargs[instrument]))

            retrievals.plot(ax=ax, color=f"C{i_color}")
            # ax.set_title(f"{gammas_labels} {shape.name} {instrument}")
            gamma.plot(ax, label=f"{'In-situ' if 'in-situ' in gammas_labels else 'Liquid'} origin", color=f"C{i_color}")
        i_color += 1

    for ax in axs:
        ax.loglog()
        ax.set_xlim(9e-6, 1e-3)
        ax.set_ylim(1e7, 1e11)
        ax.set_yticks([1e7, 1e8, 1e9, 1e10, 1e11])
        if ax == axs[1]:
            ax.set_yticklabels([])
            ax.set_ylabel("")
            ax.legend()
        else:
            ax.legend().remove()

    axs[0].set_title("Spherical")
    axs[1].set_title("Columnar")

    plt.tight_layout()
    fig.savefig("../report/img/psd_examples_2d128.pdf")
    plt.show()

    # %%
