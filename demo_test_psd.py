# %%
from psd_ast_model import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    r_vals = np.logspace(-8, -3, 100)

    psd = PSDModel(GammaPSD(1e10, 2e4, 2.5, bins=np.logspace(-8, -3, 10000)))

    fig, ax = plt.subplots()
    ax.plot(r_vals, psd.psd.psd_value(r_vals))
    # ax.set_xscale("log")
# %%

if __name__ == "__main__":
    # base_ast_model = ASTModel.from_rectangle(500, 900, 60)
    base_ast_model = ASTModel.from_diameter(1000)

    psd_doubleobs = PSDModel(GammaPSD(1e10, 2e4, 2.5, bins=np.logspace(-8, -3, 10000)), inlet_length=7.35e-3)

    hist_bins = np.linspace(0, 1e-3, 50)

    arr = psd_doubleobs.simulate_distribution_from_scaling(1000, single_particle=False, base_model=base_ast_model)
    fig, ax = plt.subplots()
    ax.hist(arr / 2 * 1e-6, bins=hist_bins)  # , bins=np.logspace(-8, -5, 100))
    plt.xlim(0, 1e-3)

    popt, pcov = fit_gamma_distribution(arr / 2 * 1e-6, hist_bins)
    n_fit = GammaPSD.n_gamma(hist_bins, *popt, 2.5)

    secax = ax.twinx()
    secax.plot(hist_bins, n_fit, label="Fit", c="red")
    # Can't get this to the right scale because we don't know the sample volume...
    # secax.plot(hist_bins, psd_doubleobs.psd.psd_value(hist_bins), label="PSD", c="purple")

    plt.legend()

    plt.show()

# %%
if __name__ == "__main__":
    psd_singleobs = PSDModel(GammaPSD(1e10, 2e4, 2.5, bins=np.logspace(-8, -3, 10000)), inlet_length=10e-2)

    hist_bins = np.linspace(0, 1e-3, 50)

    arr = psd_singleobs.simulate_distribution_from_scaling(500, single_particle=False, base_model=base_ast_model)
    fig, ax = plt.subplots()
    ax.hist(arr / 2 * 1e-6, bins=hist_bins)  # , bins=np.logspace(-8, -5, 100))
    plt.xlim(0, 1e-3)

    plt.show()

# %%
