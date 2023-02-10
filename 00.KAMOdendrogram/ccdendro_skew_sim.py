import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.stats import skewnorm

# CC(apo-apo)   Average: 0.925913, variance: 0.009192, median: 0.968700
# CC(benz-benz) Average: 0.917110, variance: 0.008674, median: 0.957200
# CC(apo-benz)  Average: 0.907780, variance: 0.008920, median: 0.947350


sample_dict=[{"name":"apo-apo",  "alpha":-10,"loc":0.98,"scale":0.05},
             {"name":"apo-benz", "alpha":-10,"loc":0.90,"scale":0.05},
             {"name":"benz-benz","alpha":-10,"loc":0.98,"scale":0.05}]

# Figure name
figname="mean85_diff_10per_absigma_0.03"

files=glob.glob("%s*"%figname)
n_exist=len(files)
if n_exist != 0:
    index=n_exist
    figname="%s_%02d"%(figname,index)

# Dendrogram title
title_s = "%s: %8.3f(%8.3f), %s:%8.3f(%8.3f), %s:%8.3f(%8.3f)" \
    % (sample_dict[0]['name'], sample_dict[0]['loc'], sample_dict[0]['scale'],
    sample_dict[1]['name'], sample_dict[1]['loc'], sample_dict[1]['scale'],
    sample_dict[2]['name'], sample_dict[2]['loc'], sample_dict[2]['scale'])

print(title_s)

def get_stat_info(cc_combination):
    for idx,s in enumerate(sample_dict):
        if s['name']==cc_combination:
            return s

def make_skew_random_cc(stat_dict):
    alpha=stat_dict['alpha']
    loc=stat_dict['loc']
    scale=stat_dict['scale']
    randcc=skewnorm.rvs(alpha, loc, scale)

    return randcc

sample_list=[]
for i in np.arange(0,100):
    sample_list.append("apo")
for i in np.arange(0,100):
    sample_list.append("benz")

dis_list = []
name_list=[]

apo_apo=[]
apo_ben=[]
ben_ben=[]

ofile=open("cc.dat","w")

for idx1,s1 in enumerate(sample_list):
    for s2 in sample_list[idx1+1:]:
        if s1=="apo" and s2=="apo":
            stat_dict=get_stat_info("apo-apo")
            name_list.append("apo-apo")
            cctmp = make_skew_random_cc(stat_dict)
            apo_apo.append(cctmp)
        elif s1=="benz" and s2=="benz":
            stat_dict=get_stat_info("benz-benz")
            name_list.append("benz-benz")
            cctmp = make_skew_random_cc(stat_dict)
            ben_ben.append(cctmp)
        else:
            stat_dict=get_stat_info("apo-benz")
            name_list.append("apo-benz")
            cctmp = make_skew_random_cc(stat_dict)
            apo_ben.append(cctmp)

        if cctmp>1.0:
            cctmp=1.0
        dist = np.sqrt(1-cctmp*cctmp)
        ofile.write("%9.5f\n"%cctmp)
        dis_list.append(dist)

ofile.close()

# Histgram of CC
fig = plt.figure(figsize=(25,10))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) #この1行を入れる
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 5])
ax1=fig.add_subplot(spec[0])
ax2=fig.add_subplot(spec[1])

#alpha=-10
#loc=0.98
#scale=0.1

# ax1.xlim(0,1)
# ax1.set_xmargin(0)
# ax1.set_ymargin(0)
ax1.set_xlim(0.70,1.0)
ax1.hist([apo_apo,apo_ben,ben_ben],bins=10,label=["apo-apo","apo-ben", "ben-ben"])
ax1.legend(loc="upper left")

#print(len(dis_list),len(name_list))
#print(apo_ben)
Z = hierarchy.linkage(dis_list, 'ward')
plt.title(title_s)

# ax2.set_xmargin(0)
# ax2.set_ymargin(0)
dn = hierarchy.dendrogram(Z,labels=sample_list, leaf_font_size=10)
# dn = hierarchy.dendrogram(Z)
plt.savefig("%s.jpg"%figname)
plt.show()

print(Z)
