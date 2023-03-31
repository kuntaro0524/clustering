import pandas as pd
import numpy as np
import sys
import pandas as pd
import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

cc_threshold = float(sys.argv[3])

cctable_path = sys.argv[1]
cctable = pd.read_csv(cctable_path, delim_whitespace=True)

# Required function
def skewed_gaussian(x, alpha, loc, scale):
    return skewnorm.pdf(x, alpha, loc, scale)
# def skewed_gaussian(x, alpha, loc, scale, base):
    # rtn_value = skewnorm.pdf(x, alpha, loc, scale) + base
    # return rtn_value

def plot_hist_and_fit(cc_data, title, color, subplot_idx):
    n_bins = int(len(cc_data) / 8)
    hist, bin_edges = np.histogram(cc_data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    initial_guess = [0, np.mean(cc_data), np.std(cc_data)]
    popt, pcov = curve_fit(skewed_gaussian, bin_centers, hist, p0=initial_guess)
    #alpha_fit, loc_fit, scale_fit, base_value = popt
    alpha_fit, loc_fit, scale_fit = popt

    plt.subplot(1, 3, subplot_idx)
    # plt.hist(cc_data, bins=n_bins, color=color, alpha=0.5, density=True)
    plt.hist(cc_data, bins=n_bins, color=color, alpha=0.5)
    plt.plot(bin_centers, skewed_gaussian(bin_centers, *popt), 'r-', label='Fitted Skew Gaussian')

    plt.title(title)
    plt.xlabel('CC')
    plt.ylabel('Frequency')
    plt.legend()

    plt.annotate(f"$\\alpha$= {alpha_fit:.2f}", xy=(0.5, 0.85), xycoords='axes fraction')
    plt.annotate(f"$\\xi$= {loc_fit:.3f}", xy=(0.5, 0.75), xycoords='axes fraction')
    plt.annotate(f"$\\sigma$= {scale_fit:.3f}", xy=(0.5, 0.65), xycoords='axes fraction')


filename_list = pd.read_csv(sys.argv[2], header=None, names=["type"])

# CCの数値が0.8以上のものだけに限定する
filter_condition = cctable['cc'] >= cc_threshold
cctable = cctable[filter_condition]

apo_apo_cc=[]
apo_benz_cc=[]
benz_benz_cc=[]


for index, row in cctable.iterrows():
    i_type = filename_list.iloc[int(row['i']), 0]
    j_type = filename_list.iloc[int(row['j']), 0]

    if i_type == 'apo' and j_type == 'apo':
        apo_apo_cc.append(row['cc'])
    elif i_type == 'apo' and j_type == 'benz':
        apo_benz_cc.append(row['cc'])
    elif i_type == 'benz' and j_type == 'benz':
        benz_benz_cc.append(row['cc'])

# Fitting to the skewed gaussian function
plt.figure(figsize=(15, 5))

plot_hist_and_fit(apo_apo_cc, "CC$_{apo.apo}$", "blue", 1)
plot_hist_and_fit(benz_benz_cc, "CC$_{benz.benz}$", "red", 2)
plot_hist_and_fit(apo_benz_cc, "CC$_{apo.benz}$", "green", 3)

plt.tight_layout()
plt.savefig("SUPFIG.png")
plt.show()