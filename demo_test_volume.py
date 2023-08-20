# %%
from matplotlib import lines
from tqdm.autonotebook import tqdm

if __name__ == "__main__":
    from cProfile import label
    from copy import deepcopy
    import logging
    from multiprocessing import process 
    from venv import logger
    from matplotlib.colors import Normalize
    from matplotlib.transforms import offset_copy

    import numpy as np
    import matplotlib.pyplot as plt
    import datetime

    from ast_model import AmplitudeField
    from psd_ast_model import GammaPSD, CrystalModel
    from cloud_model import CloudVolume, Detector
    from detector_model import Detector, DiameterSpec
    from retrieval_model import Retrieval
    from detector_run import DetectorRun

    from profiler import profile

    # %%
    logging.basicConfig(level=logging.INFO)

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["font.size"] = 10
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    

    save_figures=True

    # # %% PSD initialisation
    # # reinitialise the random seed
    # seed(42)
    # np.random.seed(42)

    # gamma_dist = GammaPSD(1.17e43, 8.31e4, 7.86)

    # fig, ax = plt.subplots()
    # gamma_dist.plot(ax)


    # # %% Cloud generation
    # # psd.plot(ax)
    # cloud_len = 100001
    # try:
    #     with open(f"../data/cloud_01_{cloud_len}_01.pkl", "rb") as f:
    #         cloud = pickle.load(f)
    # except (FileNotFoundError, ModuleNotFoundError):
    #     cloud = CloudVolume(gamma_dist, (0.01, cloud_len, 0.1))
    #     with open(f"../data/cloud_01_{cloud_len}_01.pkl", "wb") as f:
    #         pickle.dump(cloud, f)

    # print(cloud.n_particles)


    # # %% Example particle observation
    # pcle = cloud.particles.iloc[0]
    # detector_location = pcle.position - np.array([300e-6, 15*pcle.diameter, 4e-2])

    # n_pixels = 128

    # detector_1 = Detector(detector_location, n_pixels=n_pixels)

    # image = cloud.take_image(detector_1, distance=30* pcle.diameter).images[0].amplitude.intensity.field
    # plt.imshow(image)
    # plt.scatter(0, n_pixels / 2, c="r")
    # plt.colorbar()


    # %% Helper functions
    @profile(f"../data/profile__take_image__{datetime.datetime.now():%Y-%m-%d_%H%M}.prof")
    def take_image(detector, distance, cloud: CloudVolume, single_image=False, binary_output=False, **kwargs):
        return cloud.take_image(detector, distance=distance, single_image = single_image, binary_output=binary_output)

    def make_run(shape, cloud,  distance, n_px, det_len=np.inf, plot=True, px_size=10, save_run=False, offset=0, identifier=None, binary_output=False, **kwargs):
        detector_run_version=12
        cloud.set_model(shape)

        base_distance = np.max(distance)
        # if det_len == np.inf: # TODO: this is unceccesary: the DetectorRun will be the same regardless of confinemnt...
        #     identifier = "unconfined_" + identifier
        # else:
        #     identifier = f"confined_{det_len:.3e}_" + identifier
        file_name = f"run_v{detector_run_version}_{base_distance:.1f}_{n_px}px_{shape.name}_{identifier+'_' if identifier else ''}run".replace(".", "_")
        logging.info(f"Processing {file_name}")
        detector = Detector(np.array([0.005, 0.1+offset, 0.01]), n_pixels=n_px, arm_separation=0.06, detection_length=det_len, pixel_size=px_size*1e-6)
        try:
            base_run = DetectorRun.load(f"../data/{file_name}.pkl")
            base_run.detector = detector
            logging.info(f"Loaded run from file {file_name}.pkl")
        except FileNotFoundError:
            # run = cloud.take_image(detector, distance=distance, separate_particles=True)
            base_run = take_image(detector, base_distance, cloud, binary_output=binary_output)
            if save_run:
                base_run.save(f"../data/{file_name}.pkl")
                logging.info(f"Saved run to file {file_name}.pkl")


        diameter_spec = DiameterSpec(min_sep=5e-4, z_confinement=True)

        distance = [distance] if isinstance(distance, (int, float)) else distance
        
        base_retrieval = Retrieval(base_run, diameter_spec)
        retrievals = [base_retrieval.slice(run_distance) for run_distance in distance]
        # retrievals = [Retrieval(run, diameter_spec) for run in runs]
        if plot:
            [retrieval.fancy_plot(cloud, **kwargs) for retrieval in retrievals]

        return base_run, retrievals

    # # %% PSD retrieval examples - habit and 2D-S
    # for shape in [CrystalModel.SPHERE, CrystalModel.RECT_AR5]:
    # #     run, retrievals = make_run(shape, 999, 128)
    #     logging.info(f"Processing {shape.name}")
    #     logging.info("\tNo z confinement beyond arms...")
    #     run, retrievals = make_run(shape, 1000, 128, make_fit=False, plot_true_adjusted=False)
    #     logging.info("\tWith 1mm z confinement...")
    #     run, retrievals = make_run(shape, 1000, 128, det_len=128*10e-6, px_size=10, make_fit=False, plot_true_adjusted=False)
    #     logging.info("Done.")

    # # %% PSD retrieval examples - rect 2D-S, offset
    # for offset in np.linspace(0,10000, 10, endpoint=False):
    #     logging.info(f"Processing offset {offset}")
    #     run, retrievals = make_run(CrystalModel.RECT_AR5, 1000, 128, det_len=128*10e-6, px_size=10, make_fit=False, plot_true_adjusted=False, offset=offset)
    #     retrievals[-1].fancy_plot(cloud, make_fit=False, plot_true_adjusted=False)
    #     logging.info("Done.")

    # # %% plot sample volume as a function of diameter
    run_2d128_for_volume = DetectorRun(Detector(np.array([0,0,0]), 0.06), [], 1000)
    run_2ds_for_volume = DetectorRun(Detector(np.array([0,0,0]), 0.06, detection_length=128*10e-6), [], 1000)
    run_2ds_1024_for_volume = DetectorRun(Detector(np.array([0,0,0]), 0.06, detection_length=1024*10e-6, n_pixels=1024), [], 1000)

    diameters = np.linspace(8e-6, 10e-3, 1000)

    volumes_2d128 = [run_2d128_for_volume.volume(diameter) for diameter in diameters]
    volumes_2ds = [run_2ds_for_volume.volume(diameter, DiameterSpec(z_confinement=True)) for diameter in diameters]
    volumes_2ds_hybrid = [run_2d128_for_volume.volume(diameter) if diameter > 300e-6 else run_2ds_for_volume.volume(diameter, DiameterSpec(z_confinement=True)) for diameter in diameters]
    volumes_2ds_1024 = [run_2ds_1024_for_volume.volume(diameter, DiameterSpec(z_confinement=True)) for diameter in diameters]



    fig, ax = plt.subplots(figsize=(6,2))
    tr = offset_copy(ax.transData, fig=fig, x=0.2, y=-1.5, units='points')
    tr2 = offset_copy(ax.transData, fig=fig, x=-0.2, y=1.5, units='points')
    ax.plot(diameters, volumes_2d128, label="2D-128")
    ax.plot(diameters, volumes_2ds, label="2D-S", zorder=9999)
    ax.plot(diameters, volumes_2ds_hybrid, label="2D-S (hybrid)",transform=tr)
    ax.plot(diameters, volumes_2ds_1024, label="Adapted 2D-S (1024 pixels)", transform=tr2)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1.04), frameon=False)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(left=8e-6, right=10e-3)
    ax.set_ylim(1e-5)
    ax.set_ylabel("Sample volume (m$^3$ km$^{-1}$)")
    ax.set_xlabel("Diameter (µm)")
    ax.set_yticks([1e-5,1e-4,1e-3, 1e-2, 1e-1, 1])
    plt.tight_layout()
    ax.set_xticklabels([f"{1e6*tick:.0f}" for tick in ax.get_xticks()])

    if save_figures:
        plt.savefig("../report/img/sample-volume.pdf", bbox_inches="tight")
    plt.show()

    # %% Residuals for different distances and PSDs

    # shape = CrystalModel.SPHERE
    # max_len = 1e3
    # z_confinement = True
    # identifier="2d128"
    # n_px = 128
    # px_size = 10
    gammas = {
        ("in-situ", "cold"): GammaPSD.w19_parameterisation(-70, total_number_density=44, insitu_origin=True),
        ("in-situ", "hot"): GammaPSD.w19_parameterisation(-50, total_number_density=44, insitu_origin=True),
        ("liquid", "cold"): GammaPSD.w19_parameterisation(-60, total_number_density=44, liquid_origin=True),
        ("liquid", "hot"): GammaPSD.w19_parameterisation(-40, total_number_density=44, liquid_origin=True),
    }


    clouds_ensemble = []
    for i in range(1):
        gamma_clouds = {}
        for labels, gamma in gammas.items():
            gamma_clouds[labels] = CloudVolume(gamma, (0.1, 25_001, 0.1), random_seed=i)
        
        clouds_ensemble.append(gamma_clouds)

    logger.info("Calculating means and number densities...")
    for labels, gamma in gammas.items():
        logging.info(f"{labels}: n_0 {gamma.total_number_density:.2f} m-3; D_mean {gamma.mean*1e6:.2f} um")
    print()

    def process_residuals(n_pts, n_px, max_len, cloud, px_size, det_len, identifier, shape, moments=[0], binary_output=False):

        max_len_power = np.log10(max_len)
        # residuals = np.zeros((n_pts, n_px-1))
        run_distances = np.logspace(0,max_len_power, n_pts)
        run, retrievals = make_run(
            shape,
            cloud,
            run_distances, 
            n_px, 
            px_size=px_size, 
            plot=False, 
            save_run=True,
            make_fit=False,
            det_len=det_len,
            identifier=identifier,
            binary_output=binary_output
        )

        moment_residuals = {}
        if 0 not in moments:
            moments = moments + [0]

        for moment in moments:
            residuals = np.zeros((n_pts, n_px-1))
            for i, retrieval in enumerate(retrievals):
                true_psd = cloud.psd.dn_dd(retrieval.midpoints)
                residuals[i,:] =  (retrieval.dn_dd_measured - true_psd)* (retrieval.midpoints/cloud.psd.mean)**moment
            
            moment_residuals[moment] = deepcopy(residuals)

        return moment_residuals, retrievals


    def residuals_in_context(shape, z_confinement, identifier, n_px=128, px_size=10, max_len=10_000, n_pts=51, residual_labels=None, moments=[0], binary_output=True):
        det_len = n_px*px_size*1e-6 if z_confinement else np.inf
        true_psd = None
        cloud_len = max_len + 1
        max_len_power = np.log10(max_len)

        iterator = gamma_clouds.keys() if residual_labels is None else [residual_labels]

        best_retrievals = {}
        gamma_residuals = {}
        for labels in iterator:
            cloud = gamma_clouds[labels]

            moment_residuals, retrievals = process_residuals(n_pts, n_px, max_len, cloud, px_size, det_len, identifier+"-"+labels[0]+"-"+labels[1], shape, moments=moments, binary_output=binary_output)
            
            best_retrievals[labels] = deepcopy(retrievals[-1])

            for moment in moments:
                if moment != 0:
                    residual_labels = (*labels, moment)
                else:
                    residual_labels = labels
                gamma_residuals[residual_labels] = moment_residuals[moment]

            del retrievals
        
        return best_retrievals, gamma_residuals

    def plot_residuals(residuals, best_retrieval, px_size=10, n_px=128, max_len=10_000, n_pts=51):
        run_distances=np.logspace(0,np.log10(max_len),n_pts)
        
        # find the last bin with data
        last_bin = np.where(best_retrieval.dn_dd_measured != 0)[0].max()
        norm = plt.Normalize(vmin=0, vmax=last_bin)
        cmap = plt.cm.viridis
        cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        # Conventional scale
        fig, ax = plt.subplots()
        for i, x_px in tqdm(enumerate(np.linspace(px_size,px_size*(n_px+1), n_px-1))):
            if i > last_bin:
                break
            ax.plot(run_distances, residuals[:,i], label=f"{x_px:.0f} µm", color=cmap(norm(i)), linewidth=1)
        fig.colorbar(cbar, label="Diameter/pixels")
        ax.hlines([0], 0, n_pts-1, color="grey", linestyle="dotted")
        # plt.title(f"Fitting {shape.name} PSD with {n_px}x{px_size} µm pixels, with {f'{det_len:.3e}' if z_confinement else 'no'} z confinement")
        plt.ylabel("Residuals/ $\mathrm{m^{-3} m^{-1}}$")
        plt.xlabel("Distance/ m")
        # ax.set_ylim(-0.15e9,1.75e9)
        ax.set_xlim(0, run_distances[-1])
        # ax.text(0.05, 0.95, f"{labels[0]} origin, {labels[1]}", transform=ax.transAxes, verticalalignment='top')

    def plot_residual_heatmap(residuals, best_retrieval=None, px_size=10, max_len=10_000, n_pts=51, true_psd=None, normalised=False):
        run_distances=np.logspace(0,np.log10(max_len),n_pts)
        # Heatmap
        if best_retrieval is not None:
            last_bin = np.where(best_retrieval.dn_dd_measured != 0)[0].max()
        else:
            last_bin = residuals.shape[1]-1
        x_mesh = run_distances
        y_mesh = np.arange(px_size,px_size*(last_bin+1.1), px_size)

        fig, ax = plt.subplots()
        # extreme_value = np.abs(residuals).max()
        # final_iqr = np.percentile(residuals[-1,:last_bin+1], 90) - np.percentile(residuals[-1,:last_bin+1], 10)
        if normalised:
            norm = plt.cm.colors.SymLogNorm(linthresh=0.075, linscale=1, vmin=-10, vmax=10)
        else:
            linthresh = true_psd.max_dn_dd if true_psd is not None else 2e7
            norm = plt.cm.colors.SymLogNorm(linthresh=linthresh/2, linscale=1, vmin=-2e9, vmax=2e9)
        plt.pcolormesh(x_mesh, y_mesh,residuals[:, :last_bin+1].T, cmap="PiYG", norm=norm)#, extent=[0, run_distances[-1], px_size, px_size*(last_bin+1)], aspect="auto")
        plt.colorbar()
        # ax.hlines([0], 0, n_pts-1, color="grey", linestyle="dotted")
        # plt.title(f"Fitting {shape.name} PSD with {n_px}x{px_size} µm pixels, with {det_len if z_confinement else 'no'} z confinement")
        plt.ylabel("Diameter/ µm")
        plt.xlabel("Distance/ m")
        ax.set_xscale("log")
        # ax.set_ylim(-0.15e9,1.75e9)
        # ax.set_xlim(0, run_distances[-1])
        # ax.text(0.05, 0.95, f"{labels[0]} origin, {labels[1]}", transform=ax.transAxes, verticalalignment='top')
        # plt.show()

    def retrieval_plots(best_retrievals, gamma_residuals, gamma_label, moment=0):
        gamma = gamma_label
        if moment != 0:
            residuals_label = gamma_label + (moment,)
        else:
            residuals_label = gamma_label

        plot_residuals(gamma_residuals[residuals_label], best_retrievals[gamma], max_len=base_run_len, n_pts=n_pts)
        plt.title(shape_instrument+f"_{gamma[0]}-{gamma[1]}")
        plot_residual_heatmap(gamma_residuals[residuals_label], best_retrievals[gamma], max_len=base_run_len, n_pts=n_pts, true_psd=gammas[gamma])
        plt.title(shape_instrument+f"_{gamma[0]}-{gamma[1]}")
        fig, axs = best_retrievals[gamma].fancy_plot(gamma_clouds[gamma], make_fit=False, plot_true_adjusted=False)
        # axs[0].set_xscale("log")
        fig.suptitle(shape_instrument+f"_{gamma[0]}-{gamma[1]}")
        plt.show()

    # def retrieve_and_plot(shape, z_confinement, gamma_labels, sample_len=10_000, n_pts=51, n_px=128):
    #     best_retrievals, gamma_residuals = residuals_in_context(shape, z_confinement, "2d128", n_px=n_px, px_size=10, max_len=sample_len, n_pts=n_pts, residual_labels=gamma_labels, moments=[0,1,3,6])
    #     retrieval_plots(best_retrievals, gamma_residuals)

    # %%
    results_ensemble = []
    for i, gamma_clouds in enumerate(clouds_ensemble):
        results = {}
        shapes = [CrystalModel.SPHERE, CrystalModel.RECT_AR5]
        z_confinements = [True, False]

        n_pts = 201
        base_run_len=10_000 # m FIXME: change back to 10km 


        # retrieval_plots(best_retrievals, gamma_residuals)

        for shape in shapes:
            for z_confinement in z_confinements:
                # retrieve_and_plot(shape, z_confinement, ("liquid","cold"), sample_len=50_000, n_pts=101, n_px=128)
                best_retrievals, gamma_residuals = residuals_in_context(shape, z_confinement, f"2d128_repeat{i}", n_px=128, px_size=10, max_len=base_run_len, n_pts=n_pts, moments=[0,1,3,6])
                # if z_confinement==True and shape==CrystalModel.RECT_AR5:
                #     best_retrievals, gamma_residuals = residuals_in_context(shape, z_confinement, "2d128-1024" , n_px=1024, px_size=10, max_len=max_len, n_pts=n_pts, moments=[0,1,3,6])
                label = shape.name + ("_2ds" if z_confinement else "_2d128")
                results[label] = (best_retrievals, gamma_residuals)
        
        results_ensemble.append(results)

    #  %% Plotting - each PSD, each shape, each instrument
    for gamma in gammas.keys():
        for shape_instrument in results.keys():
            best_retrievals, gamma_residuals = results[shape_instrument]
            retrieval_plots(best_retrievals, gamma_residuals, gamma)


    # %% Plotting - PSDs

    fig, axs = plt.subplots(2,2, figsize=(7.2, 4.8))#, sharex=True, sharey=True)

    axs = {
        "SPHERE_2d128": axs[0,0],
        "SPHERE_2ds": axs[0,1],
        "RECT_AR5_2d128": axs[1,0],
        "RECT_AR5_2ds": axs[1,1],
    }
    letters = iter(["a", "b", "c", "d"])

    for shape_instrument in axs.keys():
        if shape_instrument not in results.keys():
            continue
        best_retrievals, gamma_residuals = results[shape_instrument]

        number = 0
        ax = axs[shape_instrument]
        for spec, gamma in gammas.items():
            if "cold" not in spec:
                number += 1
                continue
            gamma.plot(ax=ax, label=f"{spec[0]} origin, {spec[1]} cirrus\n{gamma.parameter_description()}", color=f"C{number}")
            best_retrievals[spec].plot(ax=ax, color=f"C{number}")#, fill=True, alpha=0.2)
            number += 1

        ax.get_legend().remove()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-5, 1e-3)
        ax.set_ylim(1e5, 1e11)
        title = f"{'Sphere' if shape_instrument.startswith('SPHERE') else 'Rectangular'}; {'2D-S' if shape_instrument.endswith('2ds') else '2D-128'}"
        ax.set_title(f"({next(letters)}) {title}", loc="left")

        if shape_instrument == "SPHERE_2d128":
            h, _ = ax.get_legend_handles_labels()
            print(h)
            # h = [h[0], h[2]]
            ax.legend(h, ["In situ origin", "Liquid origin"],loc="upper right")

    plt.tight_layout()
    if save_figures:
        plt.savefig("../report/img/psd_retrievals.pdf")
    plt.show()


    # %%
    # def ensemble_residuals(gamma, identifier, n_pts=51, length=10_000, shape=CrystalModel.SPHERE, z_confinement=False, n_px=128, px_size=10, n_runs=10):    
    #     det_len = n_px*px_size*1e-6 if z_confinement else np.inf
    #     collected_residuals = []
    #     for i in range(n_runs):
    #         cloud = CloudVolume(gamma, (0.01, length+1, 0.1), random_seed=i)
    #         residuals, retrievals = process_residuals(n_pts, n_px, length, cloud, px_size, det_len, f"{identifier}_{i}", shape)
    #         collected_residuals.append(residuals)
    #         del retrievals
    #     return np.mean(collected_residuals, axis=0)

    # avg_residuals = ensemble_residuals(gammas[("liquid", "cold")], "2d128_liquid-cold")
    # plot_residual_heatmap(avg_residuals)


    # %% Particle AST model examples
    from ast_model import AmplitudeField
    total_amplitude_focused = AmplitudeField(np.ones((128, 300), dtype=np.complex128))
    total_amplitude_unfocused = AmplitudeField(np.ones((128, 300), dtype=np.complex128))

    from psd_ast_model import PositionedParticle
    for i, crystal_model in enumerate([CrystalModel.SPHERE, CrystalModel.RECT_AR5, CrystalModel.ROS_6]):
        generator = crystal_model.get_generator()
        particle = PositionedParticle(200e-6, (0,0), crystal_model,np.array([0,.5e-3+1e-3*i,0]))
        ast_model = generator(particle, pixel_size=10e-6)

        total_amplitude_focused.embed(ast_model.process(0), particle, np.array([0,0,0]))
        total_amplitude_unfocused.embed(ast_model.process(0.03), particle, np.array([0,0,0]))
        

        # ast_model.process(0.1).intensity.plot(colorbar=True)
        # ast_model.process(0.1).intensity.plot(grayscale_bounds=[.35, .5, .65])
    fig, axs = plt.subplots(1, 3, figsize=(6.5,3.3), sharey=True, sharex=True,)
    total_amplitude_focused.intensity.plot(ax=axs[0])
    total_amplitude_unfocused.intensity.plot(colorbar=True, ax=axs[1])
    total_amplitude_unfocused.intensity.plot(grayscale_bounds=[.35, .5, .65], ax=axs[2], colorbar=True)
    for i in range(3):
        axs[i].set_title(f"({chr(97+i)})", loc="left")
    plt.tight_layout()
    if save_figures:
        plt.savefig("../report/img/ast_examples.pdf", bbox_inches="tight")

    # %% Check parameterisation

    ln_slope_param_iso = lambda temp: -0.06837 * temp + 3.492 #cm^-1
    shape_param_iso = lambda ln_slope: 0.02819 * np.exp(0.7216*ln_slope) 

    ln_slope_param_lo = lambda temp: 4.937 * np.exp(-0.001846*temp) #cm^-1
    shape_param_lo = lambda ln_slope: 0.001379 * np.exp(1.285*ln_slope)


    fig, (ax1, ax2)  = plt.subplots(1,2)
    temps = np.linspace(-80, 0, 100)
    ax1.plot(temps, np.exp(ln_slope_param_iso(temps)), label="insitu")
    ax1.plot(temps, np.exp(ln_slope_param_lo(temps)), label="liquid")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Slope parameter (cm$^{-1}$)")
    ax1.set_yscale("log")
    ax1.grid()

    slopes = np.linspace(100, 10000, 100)
    ax2.plot(slopes, shape_param_iso(np.log(slopes)), label="insitu")
    ax2.plot(slopes, shape_param_lo(np.log(slopes)), label="liquid")
    ax2.set_xlabel("Slope")
    ax2.set_ylabel("Shape parameter")
    ax2.set_xscale("log")
    ax2.set_ylim(-2, 21)
    ax2.grid()
    plt.tight_layout()

    ax2.legend()


    # %% Residual processing...

    # def normalise_resdiuals(gamma, moment=0):
    #     if moment != 0:
    #         residuals_label = gamma + (moment,)
    #     else:
    #         residuals_label = gamma

    #     residuals = gamma_residuals[residuals_label]
    #     psd = gammas[gamma]
    #     retrieval = best_retrievals[gamma]
    #     bin_widths = retrieval.bin_widths

    #     return bin_widths * residuals /np.sum(psd.binned_distribution * (psd.midpoints/psd.mean)**moment)

    # def resolution_and_accuracy(gamma, moment=0, make_plots=False, output_log=True):
    #     retrieval = best_retrievals[gamma]
    #     max_len_power = np.log10(base_run_len)
    #     run_distances = np.logspace(0,max_len_power, n_pts)


    #     normalised_residuals = normalise_resdiuals(gamma, moment=moment)
    #     final_residuals = normalised_residuals[-1,:]
    #     first_converged = np.where(np.mean((np.abs(normalised_residuals - final_residuals) < 0.02)[:, best_retrievals[gamma].dn_dd_measured != 0], axis=1) > 0.9)[0].min() # ???

    #     spatial_resolution = run_distances[first_converged]
    #     med_residual= np.median(np.abs(final_residuals)[best_retrievals[gamma].dn_dd_measured != 0])
    #     max_residual = np.abs(final_residuals).max()

    #     if make_plots:
    #         plot_residual_heatmap(normalised_residuals, retrieval, max_len=base_run_len, n_pts=n_pts, true_psd=gammas[gamma], normalised=True)
    #         plt.title(shape_instrument+f"_{gamma[0]}-{gamma[1]}")

    #         fig,ax = plt.subplots()
    #         plt.scatter(retrieval.midpoints[best_retrievals[gamma].dn_dd_measured != 0], final_residuals[best_retrievals[gamma].dn_dd_measured != 0])
    #         plt.show()

    #     if output_log:
    #         logging.info(f"{gamma[0]} origin-{gamma[1]}; moment {moment}: {spatial_resolution=:.2e} m, {med_residual=:.2e}, {max_residual=:.2e}")

    #     return spatial_resolution, med_residual, max_residual

    # for shape_instrument in results.keys():
    #     logging.info(f"Processing {shape_instrument}...")
    #     best_retrievals, gamma_residuals = results[shape_instrument]
    #     for gamma in best_retrievals.keys():
    #         resolution_and_accuracy(gamma, moment=0, make_plots=True)
    #         resolution_and_accuracy(gamma, moment=1)
    #         resolution_and_accuracy(gamma, moment=3)
    #         resolution_and_accuracy(gamma, moment=6)

    # %% Moments processing
    import itertools
    def agg_moment_retrieval(retrieval, distance, moment=0):
        if isinstance(distance, (int, float)):
            distance = [distance]
        
        for distance in distance:
            sliced_retrieval = retrieval.slice(distance)
            yield np.trapz(sliced_retrieval.dn_dd_measured  * (sliced_retrieval.midpoints)**moment, sliced_retrieval.midpoints)



    moments = [0,1,3]
    distances = np.logspace(0, np.log10(base_run_len), n_pts)
    
    moments_ensemble = []
    for n_repeat, results in enumerate(results_ensemble):
        retrieved_moments = {}
        retrieved_moments_pc = {}

        for id_tuple in tqdm(itertools.product(gammas.keys(),  moments)):
            gammas_labels, moment = id_tuple

            gamma = gammas[gammas_labels]
            underlying_moment = gamma.moment(moment) * np.ones_like(distances)
            pc_underlying_moment = np.ones_like(underlying_moment)

            retrieved_moments[id_tuple] = underlying_moment
            retrieved_moments_pc[id_tuple] = pc_underlying_moment
            

        for id_tuple in tqdm(itertools.product(gammas.keys(), [CrystalModel.SPHERE, CrystalModel.RECT_AR5], moments, ["2ds", "2d128"])):
            gammas_labels, shape, moment, instrument = id_tuple

            gamma = gammas[gammas_labels]
            best_retrievals, gamma_residuals = results[f"{shape.name}_{instrument}"]
            retrievals = best_retrievals[gammas_labels]

            retrieved_moments_iteration = np.array(list(agg_moment_retrieval(retrievals, distances, moment=moment)))
            pc_retrieved_moments_itertion = retrieved_moments_iteration / retrieved_moments[(gammas_labels, moment)]

            retrieved_moments[id_tuple] = retrieved_moments_iteration
            retrieved_moments_pc[id_tuple] = pc_retrieved_moments_itertion
        
        moments_ensemble.append((retrieved_moments, retrieved_moments_pc))
    
    retrieved_moments = {key: np.mean([moments_ensemble[i][0][key] for i_repeat in range(len(moments_ensemble))], axis=0) for key in moments_ensemble[0][0].keys()}
    retrieved_moments_pc = {key: np.mean([moments_ensemble[i][1][key] for i_repeat in range(len(moments_ensemble))], axis=0) for key in moments_ensemble[0][1].keys()}

            
            # [retrieved_moments[distance] for distance in distances]


            # for distance in distances:
            #     pc_retrieved_moments[distance] = {}
            #     for gamma in gammas.keys():
            #         pc_retrieved_moments[distance][gamma] = {}
            #         for n_moment in moments:
            #             pc_retrieved_moments[distance][gamma][n_moment] = agg_moment_retrieval(best_retrievals[gamma], distance, moment=n_moment)
            
            # moments_ensemble.append(pc_retrieved_moments)





    # # %%
    # moments_ensemble = []
    # for i, results in enumerate(results_ensemble):
    #     retrieved_moments = {moment:{} for moment in moments}
    #     pc_retrieved_moments_sph = {moment: {} for moment in moments}
    #     for psd_labels in gammas.keys():
    #         for n_moment in moments:
    #             retrieved_moments_moment_psd = {'sph':{}, 'col':{}}
    #             retrieved_moments_moment_psd['sph']['underlying'] = gammas[psd_labels].moment(n_moment) * np.ones_like(distances)
    #             retrieved_moments_moment_psd['sph']['2ds'] = list(agg_moment_retrieval(results[(f'SPHERE_2ds')][0][psd_labels], distances, moment=n_moment))
    #             retrieved_moments_moment_psd['sph']['2d128'] = list(agg_moment_retrieval(results[(f'SPHERE_2ds')][0][psd_labels], distances, moment=n_moment))
                
                
    #             pc_retrieved_moments_moment_psd_sph = retrieved_moments_moment_psd_sph / retrieved_moments_moment_psd_sph[0,:]
    #             retrieved_moments_sph[(*psd_labels, n_moment)] = retrieved_moments_moment_psd_sph
    #             pc_retrieved_moments_sph[(*psd_labels, n_moment)] = pc_retrieved_moments_moment_psd_sph
    #     retrieved_moments_col = {}
    #     pc_retrieved_moments_col = {}
    #     for psd_labels in gammas.keys():
    #         for n_moment in [0,1,3]:
    #             retrieved_moments_moment_psd_col = np.array([gammas[psd_labels].moment(n_moment) * np.ones_like(distances)] + [list(agg_moment_retrieval(results[(f'RECT_AR5_{instrument}')][0][psd_labels], distances, moment=n_moment)) for instrument in ['2ds', '2d128']])
    #             pc_retrieved_moments_moment_psd_col = retrieved_moments_moment_psd_col / retrieved_moments_moment_psd_col[0,:]
    #             retrieved_moments_col[(*psd_labels, n_moment)] = retrieved_moments_moment_psd_col
    #             pc_retrieved_moments_col[(*psd_labels, n_moment)] = pc_retrieved_moments_moment_psd_col
        
    #     moments_ensemble.append((pc_retrieved_moments_sph, pc_retrieved_moments_col))



    # %%
    import matplotlib.lines as mlines
    gammas_labels = list(gammas.keys())
    fig, axs = plt.subplots(len(gammas_labels),len(moments), figsize=(7.2, 5), sharey=True, sharex=True)
    # gammas_labels = [gammas_labels[0], gammas_labels[2]]


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


    #     ax = 
    # for i, psd_labels in enumerate(gammas_labels):
    #     if len(gammas_labels) == 1:
    #         gamma_axs = axs
    #     else:
    #         gamma_axs = axs[i]

    #     for ax, moment in zip(gamma_axs, [0,1,3]):
    #         for n_repeat, (pc_retrieved_moments_sph, pc_retrieved_moments_col) in enumerate(moments_ensemble):
    #             if n_repeat==0:
    #                 underlying = ax.plot(distances, pc_retrieved_moments_sph[(*psd_labels, moment)].T[:,0], color="C0")

    #             obs_2ds = ax.plot(distances, pc_retrieved_moments_sph[(*psd_labels, moment)].T[:,1], color="C1", alpha=0.3)
    #             _ = ax.plot(distances, pc_retrieved_moments_col[(*psd_labels, moment)].T[:,1], linestyle="dashed", color="C1", alpha=0.3)

    #             obs_2d128 = ax.plot(distances, pc_retrieved_moments_sph[(*psd_labels, moment)].T[:,2], color="C2", alpha=0.3)
    #             _ = ax.plot(distances, pc_retrieved_moments_col[(*psd_labels, moment)].T[:,2], linestyle="dashed", color="C2", alpha=0.3)
                
    #         # ax.set_ylim(0,10)
    #         ax.loglog()
    #         ax.set_yticks([0.1, 1, 10, 100])
    #         ax.set_xticks([1,10,100,1000,10000])
    #         ax.set_ylim(0.1, 100)
            
    #         # ax.set_yticks(range(0, 11), [f"{i}%" for i in range(0, 1100, 100)])#[0,1,2])
    #         # ax.set_yscale("lg")


    # for i, ax in enumerate(axs[0,:]):
    #     ax.set_title(f"$n_{moments[i]}$")
    
    # for i, ax in enumerate(axs[:,0]):
    #     ax.set_ylabel(f"{gammas_labels[i][0]} origin-{gammas_labels[i][1]}\nRelative error")
    
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
