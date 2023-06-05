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
    hist_bins = np.linspace(0, 1e-3, 50)

    arr = psd.simulate_distribution(1000, single_particle=False)
    fig, ax = plt.subplots()
    ax.hist(arr / 2 * 1e-6, bins=hist_bins)  # , bins=np.logspace(-8, -5, 100))
    plt.xlim(0, 1e-3)

    plt.show()

# %%
