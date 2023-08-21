# Retrieval processing: PSD fitting
# Author: Oliver Driver
# Date: 10/07/2023

import numpy as np
import matplotlib.pyplot as plt


from .cloud import CloudVolume
from .detector import  DiameterSpec
from .detector_run import DetectorRun
from .psd import GammaPSD

class Retrieval:
    def __init__(self, run: DetectorRun, spec: DiameterSpec, bins: np.array=None, slice_particles: dict=None):

        self.spec = spec
        self.run = run

        # initialise bins
        array_length = run.detector.n_pixels * run.detector.pixel_size
        self.bins = bins if bins is not None else np.linspace(0, array_length - run.detector.pixel_size, run.detector.n_pixels)

        if slice_particles is not None:
            self.detected_particles = slice_particles
        else:
            self.detected_particles = run.measure_diameters(spec) 

            if spec.z_confinement:
                y_vals = self.particles.apply(lambda row: row.position[1], axis=1)
                to_remove = []
                for loc, _ in self.detected_particles.items():
                    likely_pcle_index = np.argmin(np.abs(y_vals - loc[1]/1e6))
                    likely_pcle = self.particles.iloc[likely_pcle_index]
                    if not likely_pcle.stereo_observed:
                        to_remove.append(loc)

                for loc in to_remove:
                    self.detected_particles.pop(loc)

        self.diameters = np.array(list(self.detected_particles.values())) * 1e-6 # m

        self.midpoints = (self.bins[:-1] + self.bins[1:]) / 2
        self.bin_widths = self.bins[1:] - self.bins[:-1]

        self.volumes = run.volume(self.midpoints, spec=spec) # m^3

        self.dn_dd_measured = np.histogram(self.diameters, bins=self.bins)[0] / (self.bin_widths * self.volumes) # m^-3 m^-1


    def plot(self, ax=None, label=None, **plot_kwargs):
        ax = ax if ax is not None else plt.gca()
        ax.stairs(self.dn_dd_measured, self.bins, label=label, **plot_kwargs)
        ax.set_xlabel("Diameter (m)")
        ax.set_ylabel("dN/dD ($\mathrm{m}^{-3}\,\mathrm{m}^{-1}$)")
        ax.legend()
        return ax

    def slice(self, distance):
        distance_micron = distance * 1e6
        kept_particles = {loc:pcle_diameter for loc, pcle_diameter in self.detected_particles.items() if (distance_micron - loc[1]) > 0 }
        run = self.run.slice(distance)

        return Retrieval( run, self.spec, bins=self.bins, slice_particles=kept_particles)
    

    def fancy_plot(self, cloud:CloudVolume, make_fit=True, plot_true_adjusted=True):
        fig, axs = plt.subplots(2, 1, height_ratios=[3,1], figsize=(7.2, 5), sharex='col')

        ax = axs[0]

        true = cloud.psd.plot(ax, label=f"True\n{cloud.psd.parameter_description()}",)
        if plot_true_adjusted:
            cloud.psd.plot(ax, retrieval=self, color="C0", linestyle="dotted")
        self.plot(label="Retrieved (Circ. equiv.)", ax=ax, color="C1")
        if make_fit:
            fit = GammaPSD.fit(self.midpoints, self.dn_dd_measured, min_considered_diameter = 20e-6) # What minimum diameter is appropriate; how can we account for the low spike...
            fit_ce = fit.plot(ax, label=f"Circle equivalent\n{fit.parameter_description()}", color="C1")

        handles = true+fit_ce if make_fit else true

        ax.set_xlim(0, 5e-4)
        ax.legend(handles=handles)

        axs[1].bar(self.midpoints, np.histogram(self.diameters, bins=self.bins)[0], width=0.9*np.diff(self.bins), color="C1", alpha=0.2)
        axs[1].set_xlabel("Diameter (m)")
        axs[1].set_ylabel("Count")

        return fig, axs

    
    def remove_particles(self, locations):
        for location in locations:
            self.detected_particles.pop(location)
        self.diameters = np.array(list(self.detected_particles.values())) * 1e-6
        self.dn_dd_measured = np.histogram(self.diameters, bins=self.bins)[0] / (self.bin_widths * self.volumes)

    
    def iwc(self, as_volume=False):
        sphere_volumes = 1/6 * np.pi * self.midpoints**3 # Assumption that retrieved diamaeter is volume equivalent sphere diameter
        integrated_volume = np.sum(self.dn_dd_measured * sphere_volumes * self.bin_widths) # ∫(m^-3 m^-1)(m^3) (dm) = m^3(water) m^-3(cloud) 
        if as_volume:
            return integrated_volume # m^3(water) m^-3(cloud)
        else:
            return integrated_volume * 917 # kg(water) m^-3(cloud)
        
    @property
    def particles(self):
        if not hasattr(self.run, "particles"):
            self.run.set_particles()
        return self.run.particles
