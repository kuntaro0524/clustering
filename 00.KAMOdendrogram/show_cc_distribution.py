import pandas as pd
import numpy as np
import sys

import pandas as pd

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

nbins=50

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.hist(cc_apo_apo, bins=nbins, color='blue', alpha=0.5)
plt.title('Apo-Apo Correlation Coefficients')
plt.xlabel('CC')
plt.ylabel('Frequency')

plt.subplot(132)
plt.hist(cc_apo_benz, bins=nbins, color='green', alpha=0.5)
plt.title('Apo-Benz Correlation Coefficients')
plt.xlabel('CC')
plt.ylabel('Frequency')

plt.subplot(133)
plt.hist(cc_benz_benz, bins=nbins, color='red', alpha=0.5)
plt.title('Benz-Benz Correlation Coefficients')
plt.xlabel('CC')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("CC_distribution.png")
plt.show()