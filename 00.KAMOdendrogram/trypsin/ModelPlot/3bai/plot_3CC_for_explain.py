import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.stats import skewnorm
from scipy.cluster.hierarchy import fcluster
from scipy.stats import lognorm

# Trypsinのやつは最適値を選択している(NBINSはここに違う)
# nbins=85
aaa=[0.7044,0.0027,0.0131]
bba=[0.7648,0.0060,0.0247]
aba=[0.7871,0.0147,0.0250]

sigma_aa = aaa[0]
loc_aa = aaa[1]
scale_aa = aaa[2]

sigma_bb = bba[0]
loc_bb = bba[1]
scale_bb = bba[2]

sigma_ab = aba[0]
loc_ab = aba[1]
scale_ab = aba[2]

sample_dict=[{"name":"apo-apo","sigma":sigma_aa,"loc":loc_aa,"scale":scale_aa},
             {"name":"apo-benz","sigma":sigma_ab,"loc":loc_ab,"scale":scale_ab},
             {"name":"benz-benz","sigma":sigma_bb,"loc":loc_bb,"scale":scale_bb}]

    
# Dendrogram title
title_s = "INPUT: (%s: sigma: %8.3f loc:%8.3f scale:%8.3f)(%s sigma:%8.3f loc:%8.3f scale:%8.3f)(%s: sigma:%8.3f loc:%8.3f scale:%8.3f)" \
    % (sample_dict[0]['name'], sample_dict[0]['sigma'],sample_dict[0]['loc'], sample_dict[0]['scale'], \
    sample_dict[1]['name'], sample_dict[1]['sigma'],sample_dict[1]['loc'], sample_dict[1]['scale'], \
    sample_dict[2]['name'], sample_dict[2]['sigma'],sample_dict[2]['loc'], sample_dict[2]['scale'])

def get_stat_info(cc_combination):
    for idx,s in enumerate(sample_dict):
        if s['name']==cc_combination:
            return s

from scipy.stats import rv_continuous
from scipy.special import erf
import numpy as np

def calcCCvalue(stat_dict):
    sigma=stat_dict['sigma']
    loc=stat_dict['loc']
    scale=stat_dict['scale']
    # randccが 0~1に入るまで繰り返す
    while True:
        randcc = 1 - lognorm.rvs(sigma, loc, scale)
        if randcc >= 0.0 and randcc <= 1.0:
            break

    return randcc

# Fitted model function
x = np.arange(0.0, 1.0, 0.001)
from scipy.stats import lognorm
def log_norm(x, sigma, loc, scale):
    return lognorm.pdf(1-x, sigma, loc, scale)
# AA model
yaa = log_norm(x, sigma_aa, loc_aa, scale_aa)
# BB model
ybb = log_norm(x, sigma_bb, loc_bb, scale_bb)
# AB model
yab = log_norm(x, sigma_ab, loc_ab, scale_ab)

plt.plot(x, yaa, label="apo-apo", color="red")
plt.plot(x, ybb, label="benz-benz", color="blue")
plt.plot(x, yab, label="apo-benz", color="green")

plt.xlabel("CC")
plt.ylabel("Frequency", fontsize=30)
plt.xlim(0.9,1.0)
# yticsのフォントサイズを20ptsにする
plt.yticks(fontsize=20)
plt.xticks(fontsize=9)
#plt.annotate(f"threshold={threshold:.3f}", xy=(0.4, 0.55), xycoords='axes fraction', fontsize=20)
plt.savefig("explain.png")
plt.show()