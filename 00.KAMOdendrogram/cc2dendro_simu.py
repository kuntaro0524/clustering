import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy

# CC(apo-apo)   Average: 0.925913, variance: 0.009192, median: 0.968700
# CC(benz-benz) Average: 0.917110, variance: 0.008674, median: 0.957200
# CC(apo-benz)  Average: 0.907780, variance: 0.008920, median: 0.947350

# Figure name
aamean=float(sys.argv[1])
bbmean=float(sys.argv[2])
abmean=float(sys.argv[3])
aasig=float(sys.argv[4])
bbsig=float(sys.argv[5])
absig=float(sys.argv[6])

# N data
ndata = int(sys.argv[7])

sample_dict=[{"name":"apo-apo","mean":aamean,"sigma":aasig},
             {"name":"apo-benz", "mean":abmean,"sigma":absig},
             {"name":"benz-benz","mean":bbmean,"sigma":bbsig}]

figname="mean_aa%3.2f_sig%3.2f_bb%3.2f_sig_%3.2f_ab_%3.2f_sig_%3.2f_%03d" % \
(aamean, aasig, bbmean, bbsig, abmean, absig, ndata)

files=glob.glob("%s*"%figname)
n_exist=len(files)
if n_exist != 0:
    index=n_exist
    figname="%s_%02d"%(figname,index)

# Dendrogram title
title_s = "%s: %8.3f(%8.3f), %s:%8.3f(%8.3f), %s:%8.3f(%8.3f)" \
    % (sample_dict[0]['name'], sample_dict[0]['mean'], sample_dict[0]['sigma'],
    sample_dict[1]['name'], sample_dict[1]['mean'], sample_dict[1]['sigma'],
    sample_dict[2]['name'], sample_dict[2]['mean'], sample_dict[2]['sigma'])

print(title_s)

def get_stat_info(cc_combination):
    for idx,s in enumerate(sample_dict):
        if s['name']==cc_combination:
            return s

def make_random_cc(stat_dict):
    mean=stat_dict['mean']
    sigma=stat_dict['sigma']
    while(True):
        randcc = np.random.normal(loc=mean,scale=sigma)
        if randcc >=0 and randcc<=1.0:
            break

    return randcc

sample_list=[]
n_half = int(ndata/2)
for i in np.arange(0,n_half):
    sample_list.append("apo")
for i in np.arange(0,n_half):
    sample_list.append("benz")

dis_list = []
name_list=[]

apo_apo=[]
apo_ben=[]
ben_ben=[]

for idx1,s1 in enumerate(sample_list):
    for s2 in sample_list[idx1+1:]:
        if s1=="apo" and s2=="apo":
            stat_dict=get_stat_info("apo-apo")
            name_list.append("apo-apo")
            cctmp = make_random_cc(stat_dict)
            apo_apo.append(cctmp)
        elif s1=="benz" and s2=="benz":
            stat_dict=get_stat_info("benz-benz")
            name_list.append("benz-benz")
            cctmp = make_random_cc(stat_dict)
            ben_ben.append(cctmp)
        else:
            stat_dict=get_stat_info("apo-benz")
            name_list.append("apo-benz")
            cctmp = make_random_cc(stat_dict)
            apo_ben.append(cctmp)
        dist = np.sqrt(1-cctmp*cctmp)
        dis_list.append(dist)

# Histgram of CC
fig = plt.figure(figsize=(25,10))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) #この1行を入れる
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 5])
ax1=fig.add_subplot(spec[0])
ax2=fig.add_subplot(spec[1])

# ax1.xlim(0,1)
# ax1.set_xmargin(0)
# ax1.set_ymargin(0)
ax1.set_xlim(0.70,1.0)
ax1.hist([apo_apo,apo_ben,ben_ben],bins=10,label=["apo-apo","apo-ben", "ben-ben"])
ax1.legend(loc="upper left")

print(len(dis_list),len(name_list))
Z = hierarchy.linkage(dis_list, 'ward')
plt.title(title_s)

# ax2.set_xmargin(0)
# ax2.set_ymargin(0)
dn = hierarchy.dendrogram(Z,labels=sample_list, leaf_font_size=10)
# dn = hierarchy.dendrogram(Z)
plt.savefig("%s.jpg"%figname)
plt.show()

print(Z)
