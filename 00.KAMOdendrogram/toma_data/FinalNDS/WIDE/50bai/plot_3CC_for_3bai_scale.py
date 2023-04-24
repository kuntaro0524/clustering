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

sigma_aa,loc_aa,scale_aa=0.40937419,-0.00289777, 0.0162868
sigma_bb,loc_bb,scale_bb=1.01742228,0.00229134,0.01365336
sigma_ab,loc_ab,scale_ab=0.26456247,-0.00657071, 0.03214396


scale_aa = scale_aa*5
scale_bb = scale_bb*5
scale_ab = scale_ab*5


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
plt.xlim(0.0,1.0)
# yticsのフォントサイズを20ptsにする
plt.yticks(fontsize=20)
plt.xticks(fontsize=9)
#plt.annotate(f"threshold={threshold:.3f}", xy=(0.4, 0.55), xycoords='axes fraction', fontsize=20)
plt.savefig("explain.png")
plt.show()
