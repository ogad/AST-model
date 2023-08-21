# %%
import itertools
import logging
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from oap_model.detector_run import DetectorRun
from oap_model.psd import CrystalModel, GammaPSD
from tqdm import tqdm
from oap_model.detector import DiameterSpec, Detector
import pickle

import logging
logging.basicConfig(level=logging.INFO)

from oap_model.retrieval import Retrieval

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 10
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

gammas = {
    ("in-situ", "cold"): GammaPSD.w19_parameterisation(-70, total_number_density=44, insitu_origin=True),
    # ("in-situ", "hot"): GammaPSD.w19_parameterisation(-50, total_number_density=44, insitu_origin=True),
    ("liquid", "cold"): GammaPSD.w19_parameterisation(-60, total_number_density=44, liquid_origin=True),
    # ("liquid", "hot"): GammaPSD.w19_parameterisation(-40, total_number_density=44, liquid_origin=True),
}

detector_kwargs = {
    "2d128": {"n_pixels": 128, "arm_separation": 0.06, "pixel_size": 10e-6},
    "2ds": {"n_pixels": 128, "arm_separation": 0.06, "pixel_size": 10e-6, "detection_length": 128*10e-6},
}
diameter_spec_kwargs = {
    "2ds": {"z_confinement": True, "min_sep":5e-4},
    "2d128": {"z_confinement": False, "min_sep":5e-4},
}

base_run_len = 10000
n_pts = 500
n_repeats = 10

def agg_moment_retrieval(retrieval, distance, moment=0):
    if isinstance(distance, (int, float)):
        distance = [distance]
    
    for distance in distance:
        sliced_retrieval = retrieval.slice(distance)
        yield np.trapz(sliced_retrieval.dn_dd_measured  * (sliced_retrieval.midpoints)**moment, sliced_retrieval.midpoints)



moments = [0, 1, 3]
distances = np.logspace(0, np.log10(base_run_len), n_pts)

if __name__ == '__main__':
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
            
            logging.info("Loading...")
            base_run = DetectorRun.load(f"../data/{'-'.join(gammas_labels)}/run_{base_run_len}_{'-'.join(gammas_labels)}_128px_{shape.name}_repeat{n_repeat}.pkl")
            logging.info("Loaded!")
            base_run.trim_blank_space()
            base_run.detector = detector
            logging.info("Processing retrieval...")
            retrievals = Retrieval(base_run, DiameterSpec(**diameter_spec_kwargs[instrument]))
                                        
            logging.info("Processing slices...")
            retrieved_moments_iteration = np.array(list(agg_moment_retrieval(retrievals, distances, moment=moment)))
            pc_retrieved_moments_itertion = retrieved_moments_iteration / retrieved_moments[(gammas_labels, moment)]

            retrieved_moments[id_tuple] = retrieved_moments_iteration
            retrieved_moments_pc[id_tuple] = pc_retrieved_moments_itertion
            logging.info("Success!\n")
        
        moments_ensemble.append((retrieved_moments, retrieved_moments_pc))


    retrieved_moments = {key: np.mean([moments_ensemble[i_repeat][0][key] for i_repeat in range(len(moments_ensemble))], axis=0) for key in moments_ensemble[0][0].keys()}
    retrieved_moments_pc = {key: np.mean([moments_ensemble[i_repeat][1][key] for i_repeat in range(len(moments_ensemble))], axis=0) for key in moments_ensemble[0][1].keys()}

    # %%
    curves = list(itertools.product(gammas.keys(), [CrystalModel.SPHERE, CrystalModel.RECT_AR5], moments, ["2ds", "2d128"]))
    final_retrieved_moment_pc = {key: retrieved_moments_pc[key][-1] for key in curves}
    abs_err = {key: np.abs(retrieved_moments[key] - retrieved_moments[key][-1]) for key in curves}
    final_abs_e_fold = {key: distances[np.where(abs_err[key]/abs_err[key].max() > 0.1)[0].max()]  for key in curves}

    # final_abs_e_fold = {key: distances[np.where((np.abs(retrieved_moments_pc[key] - final_retrieved_moment_pc[key]))/(np.abs(retrieved_moments_pc[key] - final_retrieved_moment_pc[key])).max() > 0.37)[0].max()]  for key in curves}

    # %% save moments_ensemble to pickle
    import pickle
    with open(f"../data/moments_ensemble_{moments}_{len(gammas)}gammas_{n_repeats}repeats_128.pkl", "wb") as f:
        pickle.dump(moments_ensemble, f)
    # %%
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

            # notee the e-folding distance and the relative error
            errors_strings = sorted([
                f"{key[-1]}, {key[-3].name}: $d_{{0.1}}$ = {final_abs_e_fold[key]:.0f} m; $\\epsilon$ = {final_retrieved_moment_pc[key]-1:.0%}" for key in curves if key[0] == gamma_label and key[2] == moment
            ])

            ax.text(.95, .95, "\n".join(errors_strings), transform=ax.transAxes, ha="right", va="top", fontsize=8)


            # title the top row
            if i_row == 0:
                ax.set_title(f"$n_{moment}$")
            
            # label the left column
            if i_col == 0:
                ax.set_ylabel(f"{gamma_label[0]} origin-{gamma_label[1]}\nRelative value")
            
            # ax.set_ylim(0,10)
            ax.set_xlabel("Distance (m)")
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
    fig.savefig("../report/img/moments-cold.pdf", bbox_inches="tight")
    plt.show()

    # %%
