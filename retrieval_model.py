# Retrieval processing: PSD fitting
# Author: Oliver Driver
# Date: 10/07/2023

import logging

import numpy as np
import matplotlib.pyplot as plt

from psd_ast_model import GammaPSD
from detector_model import  DiameterSpec
from detector_run import DetectorRun

class Retrieval:
    def __init__(self, run: DetectorRun, spec: DiameterSpec, bins: np.array=None):

        # initialise bins
        array_length = run.detector.n_pixels * run.detector.pixel_size
        self.bins = bins if bins is not None else np.linspace(0, array_length - run.detector.pixel_size, run.detector.n_pixels)

        self.detected_particles = run.measure_diameters(spec)
        self.diameters = np.array(list(self.detected_particles.values())) * 1e-6

        self.midpoints = (self.bins[:-1] + self.bins[1:]) / 2
        self.bin_widths = self.bins[1:] - self.bins[:-1]

        self.volumes = run.volume(self.midpoints, spec=spec)

        self.dn_dd_measured = np.histogram(self.diameters, bins=self.bins)[0] / (self.bin_widths * self.volumes)


    def plot(self, ax=None, label=None, **plot_kwargs):
        ax = ax if ax is not None else plt.gca()
        ax.stairs(self.dn_dd_measured, self.bins, label=label, **plot_kwargs)
        ax.set_xlabel("Diameter (m)")
        ax.set_ylabel("dN/dD ($\mathrm{m}^{-3}\,\mathrm{m}^{-1}$)")
        ax.legend()
        return ax
    
    def fit_gamma(self, min_diameter = 50e-6):
        from scipy.optimize import curve_fit

        diameter_vals = self.midpoints[(self.dn_dd_measured != 0) & (self.midpoints >= min_diameter)]
        dn_dd_vals = self.dn_dd_measured[(self.dn_dd_measured != 0) & (self.midpoints >= min_diameter)]

        # expon = lambda d, intercept, slope: intercept * np.exp(-1 * slope * d)

        # log_gamma = lambda d, intercept, slope, shape: np.log10(GammaPSD._dn_gamma_dd(d, intercept, slope, shape))

        fit_gamma = lambda d, intercept, slope, shape: GammaPSD._dn_gamma_dd(d, intercept, slope, shape)

        results = curve_fit(GammaPSD._dn_gamma_dd, diameter_vals, dn_dd_vals / 1e6, 
                                p0=[1, 1, 1], 
                                maxfev=10000,
                                bounds=([0, 0, 1], [np.inf, np.inf, np.inf]),
                                # method='dogbox',
                                full_output=True
                               ) 
        
        intercept_l_mcb, slope, shape = results[0]
        intercept = intercept_l_mcb * 1e6

        logging.info(results[3])

        return GammaPSD(intercept, slope, shape)
    
    def remove_particles(self, locations):
        for location in locations:
            self.detected_particles.pop(location)
        self.diameters = np.array(list(self.detected_particles.values())) * 1e-6
        self.dn_dd_measured = np.histogram(self.diameters, bins=self.bins)[0] / (self.bin_widths * self.volumes)