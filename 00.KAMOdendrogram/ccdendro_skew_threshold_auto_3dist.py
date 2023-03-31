import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.stats import skewnorm
from scipy.cluster.hierarchy import fcluster

isDefault=True

if isDefault:
    # Default parameters
    # AA
    alpha_aa = -15.2177
    loc_aa = 0.9944
    scale_aa = 0.0195

    # AB
    alpha_ab = -11.3798
    loc_ab = 0.9880
    scale_ab = 0.0277

    # BB
    alpha_bb = -8.4750
    loc_bb = 0.9883
    scale_bb = 0.0251
else:
    # Large sigma & Difference simulation
    # AA
    alpha_aa = -15.2177
    loc_aa = 0.9944
    scale_aa = 0.0390

    # AB
    alpha_ab = -11.3798
    loc_ab = 0.970
    scale_ab = 0.0554

    # BB
    alpha_bb = -8.4750
    loc_bb = 0.9883
    scale_bb = 0.0502

# dictionary keeping 'skewed gaussian parameters'
sample_dict=[{"name":"A-A","alpha":alpha_aa,"loc":loc_aa,"scale":scale_aa},
             {"name":"A-B","alpha":alpha_ab,"loc":loc_ab,"scale":scale_ab},
             {"name":"B-B","alpha":alpha_bb,"loc":loc_bb,"scale":scale_bb}]

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
ofile.write("n_data,threshold,sigma\n")
n_times=int(sys.argv[1])
for ndata in np.arange(100,1001,100):
    mean,std=make_dendrogram(ndata=ndata, ntimes=n_times)
    ofile.write("%6d,%8.5f,%8.5f\n"%(ndata,mean,std))

ofile.close()

outfile=open("params.dat","w")
outfile.write("AA(alpha,loc,scale)=%12.5f %12.5f %12.5f\n"% (alpha_aa, loc_aa, scale_aa))
outfile.write("AB(alpha,loc,scale)=%12.5f %12.5f %12.5f\n"% (alpha_ab, loc_ab, scale_ab))
outfile.write("BB(alpha,loc,scale)=%12.5f %12.5f %12.5f\n"% (alpha_bb, loc_bb, scale_bb))
outfile.close()