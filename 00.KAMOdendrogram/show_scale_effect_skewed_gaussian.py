import pandas as pd
import numpy as np
import sys
import pandas as pd
import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# skewed gaussian 関数の定義
def skewed_gaussian(x, alpha, loc, scale):
    rtn_value = skewnorm.pdf(x, alpha, loc, scale)
    return rtn_value

# parameter 1
# alpha, loc, scale
# This is 'apo-apo' optimistic data
alpha = -13.851
loc = 0.97
scale = 0.0174

alpha = -10.704
loc = 0.97
scale = 0.02343

# Benz-Benz
alpha = -9.5818
loc = 0.97
scale = 0.0354


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