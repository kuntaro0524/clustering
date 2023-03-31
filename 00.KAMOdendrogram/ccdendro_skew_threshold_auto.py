import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.stats import skewnorm
from scipy.cluster.hierarchy import fcluster


# Apo-Apo distribution -> best fitting

alpha=-15.2177
scale = 0.0195
loc = 0.97

# mean value
#alpha=-11.6908
#scale = 0.0241
#loc = 0.97

# Differencial value 
delta = 0.01
cc_same = loc
cc_diff = loc - delta

sample_dict=[{"name":"A-A","alpha":alpha,"loc":cc_same,"scale":scale},
             {"name":"A-B","alpha":alpha,"loc":cc_diff,"scale":scale},
             {"name":"B-B","alpha":alpha,"loc":cc_same,"scale":scale}]

# Dendrogram title
title_s = "INPUT: (%s: alpha: %8.3f loc:%8.3f scale:%8.3f)(%s alpha:%8.3f loc:%8.3f scale:%8.3f)(%s: alpha:%8.3f loc:%8.3f scale:%8.3f)" \
    % (sample_dict[0]['name'], sample_dict[0]['alpha'],sample_dict[0]['loc'], sample_dict[0]['scale'], \
    sample_dict[1]['name'], sample_dict[1]['alpha'],sample_dict[1]['loc'], sample_dict[1]['scale'], \
    sample_dict[2]['name'], sample_dict[2]['alpha'],sample_dict[2]['loc'], sample_dict[2]['scale'])

print(title_s)

# get CC value from the designated type of distribution
def get_stat_info(cc_combination):
    for idx,s in enumerate(sample_dict):
        if s['name']==cc_combination:
            return s

# Calculation and return value of the skewed gaussian distribution
def make_skew_random_cc(stat_dict):
    alpha=stat_dict['alpha']
    loc=stat_dict['loc']
    scale=stat_dict['scale']
    randcc=skewnorm.rvs(alpha, loc, scale)

    return randcc

def make_data(ndata):
    n_each = int(ndata/2.0)

    sample_list=[]
    for i in np.arange(0,n_each):
        sample_list.append("A")
    for i in np.arange(0,n_each):
        sample_list.append("B")

    dist_list = []
    name_list=[]

    apo_apo=[]
    apo_ben=[]
    ben_ben=[]

    for idx1,s1 in enumerate(sample_list):
        for s2 in sample_list[idx1+1:]:
            if s1=="A" and s2=="A":
                stat_dict=get_stat_info("A-A")
                name_list.append("A-A")
                cctmp = make_skew_random_cc(stat_dict)
                apo_apo.append(cctmp)
            elif s1=="B" and s2=="B":
                stat_dict=get_stat_info("B-B")
                name_list.append("B-B")
                cctmp = make_skew_random_cc(stat_dict)
                ben_ben.append(cctmp)
            else:
                stat_dict=get_stat_info("A-B")
                name_list.append("A-B")
                cctmp = make_skew_random_cc(stat_dict)
                apo_ben.append(cctmp)

            if cctmp>1.0:
                cctmp=1.0
            dist = np.sqrt(1-cctmp*cctmp)
            dist_list.append(dist)
    
    return (sample_list, dist_list)

def make_dendrogram(ndata, ntimes):

    thresholds=[]
    for nnn in np.arange(0,ntimes):
    # making data
        sample_list, dist_list = make_data(ndata=ndata)

        # Histgram of CC
        fig = plt.figure(figsize=(15,10))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) #この1行を入れる
        ax1=fig.add_subplot(111)

        Z = hierarchy.linkage(dist_list, 'ward')
        dn = hierarchy.dendrogram(Z,labels=sample_list, leaf_font_size=10)
        last_merge = Z[-2]  # 最後の結合を取得
        threshold = last_merge[2]  # 最後の結合でのWard距離を取得
        thresholds.append(threshold)
        print(nnn, threshold)
        # plt.annotate(f"threshold: {threshold:.4f}", xy=(0.6, 0.65), xycoords='axes fraction')

    tha = np.array(thresholds)
    return (tha.mean(),tha.std())

#n_total=int(sys.argv[1])
# Figure name
n_total=10
prefix="alpha_%.1f_%.3f_%.3f_N%05d" % (
    sample_dict[1]['alpha'],sample_dict[1]['loc'], sample_dict[1]['scale'], n_total)

files=glob.glob("%s*"%prefix)
n_exist=len(files)
if n_exist != 0:
    index=n_exist
    prefix="%s_%02d"%(prefix,index)

ofile=open("%s.csv"%prefix,"w")

for ndata in np.arange(100,1001,100):
    mean,std=make_dendrogram(ndata=ndata, ntimes=10)
    ofile.write("%6d %8.5f %8.5f\n"%(ndata,mean,std))

ofile.close()
# plt.savefig("%s.jpg"%figname)
# plt.show()