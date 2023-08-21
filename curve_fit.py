# %%
import pandas as pd
from oap_model.psd import GammaPSD
from scipy.optimize import curve_fit
import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv("size_dist_example.csv", sep="\t")

# convert size from µm to m
df["size"] *= 1e-6

# convert dn_dd from L-1 µm-1 to L-1 m-1
df["dn_dd"] *= 1e6

popt, pcov = curve_fit(
        lambda d, intercept, slope, shape: GammaPSD._dn_gamma_dd(d, intercept, slope, shape), 
        df["size"], df["dn_dd"],p0=[1, 1, 1], maxfev=10000, bounds=([0, 0, 0], [np.inf, np.inf, 9]))


fig, ax = plt.subplots()
ax.scatter(df["size"], df["dn_dd"], label="Data")
ax.plot(df["size"], GammaPSD._dn_gamma_dd(df["size"], *popt), label="Fit")
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Size (m)")
ax.set_ylabel("dn_dd (L-1 m-1)")
ax.legend()
ax.text(1.6e-4, 7e4, f"Intercept: {popt[0]:.2e} L-1 m-1\nSlope: {popt[1]:.2e} m-1\nShape: {popt[2]:.2f}")
plt.show()


# %%
popt
# %%
