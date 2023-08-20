# %%
import itertools
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from detector_run import DetectorRun
from psd_ast_model import CrystalModel
from tqdm import tqdm
from detector_model import DiameterSpec, Detector


from repeat_run_hpc import gammas

from retrieval_model import Retrieval

detector_kwargs = {
    "2d128": {"n_pixels": 128, "arm_separation": 0.06, "pixel_size": 10e-6},
    "2ds": {"n_pixels": 128, "arm_separation": 0.06, "pixel_size": 10e-6, "detection_length": 128*10e-6},
}
diameter_spec_kwargs = {
    "2ds": {"z_confinement": True},
    "2d128": {"z_confinement": False},
}

base_run_len = 100
n_pts = 1000
n_repeats = 1

def agg_moment_retrieval(retrieval, distance, moment=0):
    if isinstance(distance, (int, float)):
        distance = [distance]
    
    for distance in distance:
        sliced_retrieval = retrieval.slice(distance)
        yield np.trapz(sliced_retrieval.dn_dd_measured  * (sliced_retrieval.midpoints)**moment, sliced_retrieval.midpoints)



moments = [0,1,3]
distances = np.logspace(0, np.log10(base_run_len), n_pts)

moments_ensemble = []
for n_repeat in range(n_repeats):
    retrieved_moments = {}
    retrieved_moments_pc = {}

    # get underlying moments
    for id_tuple in tqdm(itertools.product(gammas.keys(),  moments)):
        gammas_labels, moment = id_tuple

        gamma = gammas[gammas_labels]
        underlying_moment = gamma.moment(moment) * np.ones_like(distances)
        pc_underlying_moment = np.ones_like(underlying_moment)

        retrieved_moments[id_tuple] = underlying_moment
        retrieved_moments_pc[id_tuple] = pc_underlying_moment
        
    # get retrieval moments
    for id_tuple in tqdm(itertools.product(gammas.keys(), [CrystalModel.SPHERE, CrystalModel.RECT_AR5], moments, ["2ds", "2d128"])):
        gammas_labels, shape, moment, instrument = id_tuple

        detector = Detector(np.array([0.05, 0.1, 0.01]), **detector_kwargs[instrument])

        gamma = gammas[gammas_labels]

        base_run = DetectorRun.load(f"../data/run_{base_run_len}_{'-'.join(gammas_labels)}_128px_{shape.name}_repeat{n_repeat}.pkl")
        base_run.detector = detector
        retrievals = Retrieval(base_run, DiameterSpec(**diameter_spec_kwargs[instrument]))

        retrieved_moments_iteration = np.array(list(agg_moment_retrieval(retrievals, distances, moment=moment)))
        pc_retrieved_moments_itertion = retrieved_moments_iteration / retrieved_moments[(gammas_labels, moment)]

        retrieved_moments[id_tuple] = retrieved_moments_iteration
        retrieved_moments_pc[id_tuple] = pc_retrieved_moments_itertion
    
    moments_ensemble.append((retrieved_moments, retrieved_moments_pc))

retrieved_moments = {key: np.mean([moments_ensemble[i_repeat][0][key] for i_repeat in range(len(moments_ensemble))], axis=0) for key in moments_ensemble[0][0].keys()}
retrieved_moments_pc = {key: np.mean([moments_ensemble[i_repeat][1][key] for i_repeat in range(len(moments_ensemble))], axis=0) for key in moments_ensemble[0][1].keys()}



gammas_labels = list(gammas.keys())
fig, axs = plt.subplots(len(gammas_labels),len(moments), figsize=(7.2, 5), sharey=True, sharex=True)


for i_row, gamma_label in enumerate(gammas_labels):
    for i_col, moment in enumerate(moments):
        if len(gammas_labels) == 1:
            ax = axs[i_col]
        else:
            ax = axs[i_row, i_col]
        
        # plot underlying
        ax.plot(distances, retrieved_moments_pc[(gamma_label, moment)], color="C0")

        # plot spherical
        ax.plot(distances, retrieved_moments_pc[(gamma_label, CrystalModel.SPHERE, moment, "2ds")], color="C1")
        ax.plot(distances, retrieved_moments_pc[(gamma_label, CrystalModel.SPHERE, moment, "2d128")], color="C2")

        # plot columnar
        ax.plot(distances, retrieved_moments_pc[(gamma_label, CrystalModel.RECT_AR5, moment, "2ds")], linestyle="dashed", color="C1")
        ax.plot(distances, retrieved_moments_pc[(gamma_label, CrystalModel.RECT_AR5, moment, "2d128")], linestyle="dashed", color="C2")

        # title the top row
        if i_row == 0:
            ax.set_title(f"$n_{moment}$")
        
        # label the left column
        if i_col == 0:
            ax.set_ylabel(f"{gamma_label[0]} origin-{gamma_label[1]}\nRelative error")
        
        # ax.set_ylim(0,10)
        ax.loglog()
        ax.set_yticks([0.1, 1, 10, 100])
        # ax.set_xticks([1,10,100,1000,10000])
        ax.set_xlim(1, base_run_len)
        ax.set_ylim(0.1, 100)


plt.figlegend([
    mlines.Line2D([], [], color="C0", label="Underlying"),
    mlines.Line2D([], [], color="C1", label="2D-S"),
    mlines.Line2D([], [], color="C2", label="2D-128"),
    mlines.Line2D([], [], color="black", label="Spherical"),
    mlines.Line2D([], [], linestyle="dashed", color="black", label="Columnar"),
], ['Underlying', '2D-S', '2D-128', 'Spherical', 'Columnar'], loc="upper left", ncols=2, bbox_to_anchor=(0,0), frameon=False)#, loc= "upper left", frameon=False, bbox_to_anchor=(0, -0.2), ncol=3)
plt.tight_layout()
plt.show()

# %%
