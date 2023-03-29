import pandas as pd
import numpy as np
import sys
import pandas as pd
import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

cctable_path = sys.argv[1]
cctable = pd.read_csv(cctable_path, delim_whitespace=True)

# Required function
def skewed_gaussian(x, alpha, loc, scale):
    return skewnorm.pdf(x, alpha, loc, scale)

def plot_hist_and_fit(cc_data, title, color, subplot_idx):
    n_bins=80
    hist, bin_edges = np.histogram(cc_data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    initial_guess = [0, np.mean(cc_data), np.std(cc_data)]
    popt, pcov = curve_fit(skewed_gaussian, bin_centers, hist, p0=initial_guess)
    alpha_fit, loc_fit, scale_fit = popt

    plt.subplot(1, 3, subplot_idx)
    plt.hist(cc_data, bins=45, color=color, alpha=0.5, density=True)
    plt.plot(bin_centers, skewed_gaussian(bin_centers, *popt), 'r-', label='Fitted Skew Gaussian')
    plt.title(title)
    plt.xlabel('CC')
    plt.ylabel('Density')
    plt.legend()

    plt.annotate(f"Alpha: {alpha_fit:.4f}", xy=(0.6, 0.85), xycoords='axes fraction')
    plt.annotate(f"Loc: {loc_fit:.4f}", xy=(0.6, 0.75), xycoords='axes fraction')
    plt.annotate(f"Scale: {scale_fit:.4f}", xy=(0.6, 0.65), xycoords='axes fraction')


filename_list = pd.read_csv(sys.argv[2], header=None, names=["type"])
# filename_list = pd.read_csv(filename_list_path, header=None)

print(filename_list)

# CCの数値が0.8以上のものだけに限定する
filter_condition = cctable['cc'] >= 0.900
cctable = cctable[filter_condition]

# Data grouping
# CCデータを種類ごとに分ける
# apo_apo_cc = cctable.loc[(filename_list.iloc[cctable['i']]['type'] == 'apo') & (filename_list.iloc[cctable['j']]['type'] == 'apo'), 'cc']
# apo_benz_cc = cctable.loc[(filename_list.iloc[cctable['i']]['type'] == 'apo') & (filename_list.iloc[cctable['j']]['type'] == 'benz'), 'cc']
# benz_benz_cc = cctable.loc[(filename_list.iloc[cctable['i']]['type'] == 'benz') & (filename_list.iloc[cctable['j']]['type'] == 'benz'), 'cc']

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

print(apo_apo_cc)

# Fitting to the skewed gaussian function
plt.figure(figsize=(15, 5))
plot_hist_and_fit(apo_apo_cc, "Apo-Apo CC", "blue", 1)
plot_hist_and_fit(apo_benz_cc, "Apo-Benz CC", "green", 2)
plot_hist_and_fit(benz_benz_cc, "Benz-Benz CC ", "red", 3)

plt.tight_layout()
plt.show()