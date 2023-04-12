import pandas as pd
import numpy as np
import sys
import pandas as pd
import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

cctable_path = 'cctable.dat'
cctable = pd.read_csv(cctable_path, delim_whitespace=True)

# Input parameters
cc_threshold=float(sys.argv[1])
# ana_type: ["AA", "BB", "AB"]
ana_type = (sys.argv[2])

if ana_type!="AA" and ana_type!="BB" and ana_type!="AB":
    print("Error!! AB or BB or AB")

# CCの数値が0.8以上のものだけに限定する
filter_condition = cctable['cc'] >= cc_threshold

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

if ana_type=="AA":
    # CCの標準偏差を計算する
    sigma_data = np.std(cc_apo_apo)
    # CC data array
    ccdata = cc_apo_apo
    n_bins = int(len(ccdata) / 8)
elif ana_type=="AB":
    # CCの標準偏差を計算する
    sigma_data = np.std(cc_apo_benz)
    # CC data array
    ccdata = cc_apo_benz
    n_bins = int(len(ccdata) / 8)
elif ana_type=="BB":
    # CCの標準偏差を計算する
    sigma_data = np.std(cc_benz_benz)
    # CC data array
    ccdata = cc_benz_benz
    n_bins = int(len(ccdata) / 8)

print(f"LENGTH={len(ccdata):d}")
# 初期値？
hist, bin_edges = np.histogram(ccdata, bins=n_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# skewed gaussian 関数の定義
def skewed_gaussian(x, alpha, loc, scale):
    rtn_value = skewnorm.pdf(x, alpha, loc, scale)
    return rtn_value

# 初期値の設定と最小二乗法の計算
initial_guess = [0, np.mean(ccdata), np.std(ccdata)]
popt, pcov = curve_fit(skewed_gaussian, bin_centers, hist, p0=initial_guess)
alpha_fit, loc_fit, scale_fit = popt

# Histogram drawing
plt.hist(ccdata, bins=n_bins, color='green', alpha=0.5)
logstr=f"{ana_type:20s} Correlation Coefficitents"
plt.title(logstr)
plt.xlabel('CC')
plt.ylabel('Frequency')

# Plotting the resultant skewed gaussian function
plt.plot(bin_centers, skewed_gaussian(bin_centers, *popt), 'r-', label='Fitted Skew Gaussian')

plt.annotate(f"Alpha: {alpha_fit:.4f}", xy=(0.6, 0.85), xycoords='axes fraction')
plt.annotate(f"Loc: {loc_fit:.4f}", xy=(0.6, 0.75), xycoords='axes fraction')
plt.annotate(f"Scale: {scale_fit:.4f}", xy=(0.6, 0.65), xycoords='axes fraction')
    
plt.tight_layout()
plt.savefig("skewed_fitting_%s.png"%ana_type)
plt.legend()
plt.show()
