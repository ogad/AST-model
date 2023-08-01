# %%
import logging 
from random import seed
import pickle
from typing import is_typeddict

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime

from ast_model import AmplitudeField, plot_outline
from psd_ast_model import GammaPSD, TwoMomentGammaPSD, CrystalModel
from cloud_model import CloudVolume, Detector
from detector_model import Detector, ImagedRegion, ImageFilter, DiameterSpec
from retrieval_model import Retrieval
from detector_run import DetectorRun

from profiler import profile

logging.basicConfig(level=logging.INFO)


# %% From 
gammas = {
    ("in-situ", "cold"): GammaPSD.w19_parameterisation(-70, 2e37, insitu_origin=True),
    ("in-situ", "hot"): GammaPSD.w19_parameterisation(-50, 5e14, insitu_origin=True),
    ("liquid", "cold"): GammaPSD.w19_parameterisation(-60, 3e12, liquid_origin=True),
    ("liquid", "hot"): GammaPSD.w19_parameterisation(-40, 9e10, liquid_origin=True),
}
fig, ax = plt.subplots()
for spec, gamma in gammas.items():
    gamma.plot(ax=ax, label=f"{spec[0]} origin, {spec[1]} cirrus\n{gamma.parameter_description()}")
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-5, 1e-3)
ax.set_ylim(1e5, 1e11)
# ax.set_ylim(3e9, 1e11)

for gamma in gammas.values():
    print(gamma.total_number_density, gamma.mean*1e6)
# %% Retry with a more realistic gamma distribution
# reinitialise the random seed
seed(42)
np.random.seed(42)

gamma_dist = GammaPSD(1.17e43, 8.31e4, 7.86)

fig, ax = plt.subplots()
gamma_dist.plot(ax)
# %% Cloud generation
# psd.plot(ax)
cloud_len = 100001
try:
    with open(f"../data/cloud_01_{cloud_len}_01.pkl", "rb") as f:
        cloud = pickle.load(f)
except (FileNotFoundError, ModuleNotFoundError):
    cloud = CloudVolume(gamma_dist, (0.01, cloud_len, 0.1))
    with open(f"../data/cloud_01_{cloud_len}_01.pkl", "wb") as f:
        pickle.dump(cloud, f)

print(cloud.n_particles)

gamma_clouds = {}
for labels, gamma in gammas.items():
    gamma_clouds[labels] = CloudVolume(gamma, (0.01, cloud_len, 0.1))

# %% Example particle observation
pcle = cloud.particles.iloc[0]
detector_location = pcle.position - np.array([300e-6, 15*pcle.diameter, 4e-2])

n_pixels = 128

detector_1 = Detector(detector_location, n_pixels=n_pixels)

image = cloud.take_image(detector_1, distance=30* pcle.diameter).images[0].amplitude.intensity.field
plt.imshow(image)
plt.scatter(0, n_pixels / 2, c="r")
plt.colorbar()


# %%
@profile(f"../data/profile__take_image__{datetime.datetime.now():%Y-%m-%d_%H%M}.prof")
def take_image(detector, distance, cloud: CloudVolume, single_image=False):
    return cloud.take_image(detector, distance=distance, single_image = single_image)



def make_run(shape, distance, n_px, det_len=np.inf, plot=True, px_size=10, save_run=False, **kwargs):
    detector_run_version=5
    cloud.set_model(shape)

    base_distance = np.max(distance)
    try:
        base_run = DetectorRun.load(f"../data/run_v{detector_run_version}_{base_distance}_{n_px}px_{shape.name}_{det_len}_run.pkl")
    except FileNotFoundError:
        detector = Detector(np.array([0.005, 0.1, 0.01]), n_pixels=n_px, arm_separation=0.06, detection_length=det_len, pixel_size=px_size*1e-6)
        # run = cloud.take_image(detector, distance=distance, separate_particles=True)
        base_run = take_image(detector, base_distance, cloud)
        if save_run:
            base_run.save(f"../data/run_v{detector_run_version}_{distance}_{n_px}px_{shape.name}_run.pkl")

    diameter_spec = DiameterSpec(min_sep=5e-4, z_confinement=True)

    if isinstance(distance, (int, float)):
        if plot:
            fig, retrievals = make_and_plot_retrievals(base_run, **kwargs)
            fig.suptitle(f"{shape.__str__()}\n{n_px}x{base_run.detector.pixel_size*1e6:.0f} µm pixels, {distance} m distance")
        else:
            retrievals = (Retrieval(base_run, diameter_spec))
    else:
        if plot:
            raise NotImplementedError("Plotting for multiple distances not implemented.")
        runs = [base_run.slice(run_distance) for run_distance in distance]
        retrievals = [Retrieval(run, diameter_spec) for run in runs]

    return base_run, retrievals

# detections.amplitude.intensity.plot()
@profile(f"../data/profile__make_and_plot_retrievals__{datetime.datetime.now():%Y-%m-%d_%H%M}.prof")
def make_and_plot_retrievals(run, make_fit=True):
    # retrieval2 = Retrieval(run, DiameterSpec(diameter_method="xy", min_sep=5e-4, filled=True))
    retrieval = Retrieval(run, DiameterSpec(min_sep=5e-4, z_confinement=True))

    # retrieval = Retrieval(run, DiameterSpec(diameter_method="xy", min_sep=0.1, filled=True))
    # fit2 = GammaPSD.fit(retrieval2.midpoints, retrieval2.dn_dd_measured, min_considered_diameter = 20e-6)

    fig, axs = plt.subplots(2, 1, height_ratios=[3,1], figsize=(7.2, 5), sharex='col')

    ax = axs[0]

    true = cloud.psd.plot(ax, label=f"True\n{gamma_dist.parameter_description()}")
    cloud.psd.plot(ax, label=f"True\n{gamma_dist.parameter_description()}", retrieval=retrieval, color="C0", linestyle="dotted")
    # cloud.psd.plot(ax, label=f"True\n{gamma_dist.parameter_description()}", retrieval=retrieval2, color="C2", linestyle="dotted")
    retrieval.plot(label="Retrieved (Circ. equiv.)", ax=ax, color="C1")
    # retrieval2.plot(label="Retrieved (XY)", ax=ax, color="C2")
    if make_fit:
        fit = GammaPSD.fit(retrieval.midpoints, retrieval.dn_dd_measured, min_considered_diameter = 20e-6) # What minimum diameter is appropriate; how can we account for the low spike...
        fit_ce = fit.plot(ax, label=f"Circle equivalent\n{fit.parameter_description()}", color="C1")
    # fit_xy = fit2.plot(ax, label=f"XY mean\n{fit2.parameter_description()}", color="C2")

    # plt.yscale("log")
    # psd_axs[1].set_ylim(0, 0.5e9)
    handles = true+fit_ce if make_fit else true

    ax.set_xlim(0, 5e-4)
    ax.legend(handles=handles)

    axs[1].bar(retrieval.midpoints, np.histogram(retrieval.diameters, bins=retrieval.bins)[0], width=0.9*np.diff(retrieval.bins), color="C1", alpha=0.2)
    # axs[1][1].bar(retrieval2.midpoints, np.histogram(retrieval2.diameters, bins=retrieval.bins)[0], width=0.9*np.diff(retrieval.bins), color="C2", alpha=0.2)
    axs[1].set_xlabel("Diameter (m)")
    axs[1].set_ylabel("Count")

    # plt.tight_layout()

    return fig, (retrieval)#, retrieval2)



# %% PSD retrieval examples
for shape in [CrystalModel.SPHERE, CrystalModel.RECT_AR5]:
#     run, retrievals = make_run(shape, 999, 128)
    logging.info(f"Processing {shape.name}")
    logging.info("\tNo z confinement beyond arms...")
    run, retrievals = make_run(shape, 1000, 128)
    logging.info("\tWith 1mm z confinement...")
    run, retrievals = make_run(shape, 1000, 128, det_len=128*10e-6, px_size=10)
    logging.info("Done.")
# %%
# plot sample volume as a function of diameter
diameters = np.linspace(10e-6, 5e-4, 50)
volumes = [run.volume(diameter) for diameter in diameters]
plt.plot(diameters, volumes)


# %% Residual plotting

shape = CrystalModel.RECT_AR5
n_px = 128
px_size = 10
n_pts = 51
z_confinement = n_px*px_size*1e-6
true_psd = None


# %% # run, retrieval = make_run(CrystalModel.SPHERE, 10_000, n_px, det_len=n_px*px_size*1e-6, px_size=px_size, plot=True, save_run=False, make_fit=False)

gamma_residuals = {}
for labels, cloud in gamma_clouds.items():
    residuals = np.zeros((n_pts, n_px-1))
    run_distances = np.logspace(0,4, n_pts)
    _, retrievals = make_run(
        shape, 
        run_distances, 
        n_px, 
        px_size=px_size, 
        plot=False, 
        save_run=False, 
        make_fit=False,
        # det_len=z_confinement,
    )

    for i, retrieval in enumerate(retrievals):
        true_psd = cloud.psd.dn_dd(retrieval.midpoints) if true_psd is None else true_psd
        residuals[i,:] =  retrieval.dn_dd_measured - true_psd
    
    gamma_residuals[labels] = residuals

    fig, ax = plt.subplots()
    for i, x_px in tqdm(enumerate(np.linspace(px_size,px_size*(n_px+1), n_px-1))):
        ax.plot(run_distances, residuals[:,i], label=f"{x_px:.0f} µm", color="gray", linewidth=0.2)
    ax.hlines([0], 0, n_pts-1, color="grey", linestyle="dashed")
    plt.title(f"Fitting {shape.name} PSD with {n_px}x{px_size} µm pixels, with no z confinement")
    plt.ylabel("Residuals/ $\mathrm{m^{-3} m^{-1}}$")
    plt.xlabel("Distance/ m")

    ax.set_ylim(-0.15e9,1.75e9)
    ax.set_xlim(0, run_distances[-1])
    ax.text(0.05, 0.95, f"{labels[0]} origin, {labels[1]}", transform=ax.transAxes, verticalalignment='top')

# %%
from ast_model import AmplitudeField
total_amplitude_focused = AmplitudeField(np.ones((400, 1200), dtype=np.complex128))
total_amplitude_unfocused = AmplitudeField(np.ones((400, 1200), dtype=np.complex128))

from psd_ast_model import PositionedParticle
for i, crystal_model in enumerate([CrystalModel.SPHERE, CrystalModel.RECT_AR5, CrystalModel.ROS_6]):
    generator = crystal_model.get_generator()
    particle = PositionedParticle(500e-6, (0,0), crystal_model,np.array([0,2e-3+4e-3*i,0]))
    ast_model = generator(particle, pixel_size=10e-6)

    total_amplitude_focused.embed(ast_model.process(0), particle, np.array([0,0,0]))
    total_amplitude_unfocused.embed(ast_model.process(0.2), particle, np.array([0,0,0]))
    

    # ast_model.process(0.1).intensity.plot(colorbar=True)
    # ast_model.process(0.1).intensity.plot(grayscale_bounds=[.35, .5, .65])
fig, axs = plt.subplots(1, 3, figsize=(7,4.5), sharey=True, sharex=True,)
total_amplitude_focused.intensity.plot(ax=axs[0])
total_amplitude_unfocused.intensity.plot(colorbar=True, ax=axs[1])
total_amplitude_unfocused.intensity.plot(grayscale_bounds=[.35, .5, .65], ax=axs[2])
plt.tight_layout()
# %%


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


# %%
