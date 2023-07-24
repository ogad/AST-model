# Retrieval processing: PSD fitting
# Author: Oliver Driver
# Date: 10/07/2023

import logging 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ast_model import ASTModel


from detector_model import  DiameterSpec
from detector_run import DetectorRun

class Retrieval:
    def __init__(self, run: DetectorRun, spec: DiameterSpec, bins: np.array=None):

        self.spec = spec
        self.run = run

        # initialise bins
        array_length = run.detector.n_pixels * run.detector.pixel_size
        self.bins = bins if bins is not None else np.linspace(0, array_length - run.detector.pixel_size, run.detector.n_pixels)

        self.detected_particles = run.measure_diameters(spec) 

        if spec.z_confinement:
            to_remove = []
            for loc, _ in self.detected_particles.items():
                y_vals = self.particles.apply(lambda row: row.position[1], axis=1)
                likely_pcle_index = np.argmin(np.abs(y_vals - loc[1]/1e6))
                likely_pcle = self.particles.iloc[likely_pcle_index]
                if not likely_pcle.in_z_limits:
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
        return pd.concat([image.particles for image in self.run.images])
