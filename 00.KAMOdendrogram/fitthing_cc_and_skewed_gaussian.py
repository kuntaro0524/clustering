import pandas as pd
import numpy as np
import sys
import pandas as pd
import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import curve_fit

cctable_path = 'cctable.dat'
cctable = pd.read_csv(cctable_path, delim_whitespace=True)

# CCの数値が0.8以上のものだけに限定する
filter_condition = cctable['cc'] >= 0.8

cctable = cctable[filter_condition]

filename_list_path = 'filenames.lst'
filename_list = pd.read_csv(filename_list_path, header=None)

cc_apo_apo = []
cc_apo_benz = []
cc_benz_benz = []

for index, row in cctable.iterrows():
    i_type = filename_list.iloc[int(row['i']), 0]
    j_type = filename_list.iloc[int(row['j']), 0]

    if i_type == 'apo' and j_type == 'apo':
        cc_apo_apo.append(row['cc'])
    elif i_type == 'apo' and j_type == 'benz':
        cc_apo_benz.append(row['cc'])
    elif i_type == 'benz' and j_type == 'benz':
        cc_benz_benz.append(row['cc'])

import matplotlib.pyplot as plt

nbins=100

# CCの標準偏差を計算する
sigma_apo_beaz = np.std(cc_benz_benz)

# 初期値？
hist, bin_edges = np.histogram(cc_benz_benz, bins=nbins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# skewed gaussian 関数の定義
def skewed_gaussian(x, alpha, loc, scale):
    rtn_value = skewnorm.pdf(x, alpha, loc, scale)
    return rtn_value

# 初期値の設定と最小二乗法の計算
initial_guess = [0, np.mean(cc_benz_benz), np.std(cc_benz_benz)]
popt, pcov = curve_fit(skewed_gaussian, bin_centers, hist, p0=initial_guess)
alpha_fit, loc_fit, scale_fit = popt

# Histogram drawing
plt.hist(cc_benz_benz, bins=nbins, color='green', alpha=0.5)
plt.title('Apo-Benz Correlation Coefficients')
plt.xlabel('CC')
plt.ylabel('Frequency')

# Plotting the resultant skewed gaussian function
plt.plot(bin_centers, skewed_gaussian(bin_centers, *popt), 'r-', label='Fitted Skew Gaussian')

print(f"Skewed Gaussian parameters:")
print(f"Alpha: {alpha_fit:.4f}")
print(f"Loc: {loc_fit:.4f}")
print(f"Scale: {scale_fit:.4f}")
print(f"Sigma: {sigma_apo_beaz:.4f}")

plt.tight_layout()
plt.savefig("skewed_fitting.png")
plt.legend()
plt.show()