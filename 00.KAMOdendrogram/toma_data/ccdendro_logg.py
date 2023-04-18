import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.stats import skewnorm
from scipy.cluster.hierarchy import fcluster
from scipy.stats import lognorm

#scale=float(sys.argv[1])
n_total=int(sys.argv[1])
n_each = int(n_total/2.0)


# nbin=20, cc>=0.8
# [ 0.25706748 -0.00582829  0.01807159]
# [0.97068976 0.00463379 0.00579725]
# [ 0.24025039 -0.00899155  0.03458903]

sigma_aa,loc_aa,scale_aa= 0.25706748,-0.00582829, 0.01807159
sigma_bb,loc_bb,scale_bb= 0.97068976,0.00463379,0.00579725
sigma_ab,loc_ab,scale_ab= 0.24025039,-0.00899155, 0.03458903

sigma_aa,loc_aa,scale_aa= 0.48318638,-0.00173054, 0.01571608
sigma_bb,loc_bb,scale_bb=6.54281090e-01,4.03679076e-04,1.19122440e-02
sigma_ab,loc_ab,scale_ab= 0.41541101,-0.00301814 ,0.03127803


sample_dict=[{"name":"A-A","sigma":sigma_aa,"loc":loc_aa,"scale":scale_aa},
             {"name":"A-B","sigma":sigma_ab,"loc":loc_ab,"scale":scale_ab},
             {"name":"B-B","sigma":sigma_bb,"loc":loc_bb,"scale":scale_bb}]

# Figure name
figname="sigma_%.1f_%.3f_%.3f_N%05d" % (
    sample_dict[1]['sigma'],sample_dict[1]['loc'], sample_dict[1]['scale'], n_total)
    

files=glob.glob("%s*"%figname)
n_exist=len(files)
if n_exist != 0:
    index=n_exist
    figname="%s_%02d"%(figname,index)

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

sample_list=[]
for i in np.arange(0,n_each):
    sample_list.append("A")
for i in np.arange(0,n_each):
    sample_list.append("B")

dis_list = []
name_list=[]

apo_apo=[]
apo_ben=[]
ben_ben=[]

ofile=open("cc.dat","w")

for idx1,s1 in enumerate(sample_list):
    for s2 in sample_list[idx1+1:]:
        if s1=="A" and s2=="A":
            stat_dict=get_stat_info("A-A")
            name_list.append("A-A")
            cctmp = calcCCvalue(stat_dict)
            apo_apo.append(cctmp)
        elif s1=="B" and s2=="B":
            stat_dict=get_stat_info("B-B")
            name_list.append("B-B")
            cctmp = calcCCvalue(stat_dict)
            ben_ben.append(cctmp)
        else:
            stat_dict=get_stat_info("A-B")
            name_list.append("A-B")
            cctmp = calcCCvalue(stat_dict)
            apo_ben.append(cctmp)

        if cctmp>1.0:
            cctmp=1.0

        # cctmpが
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

# CC stats
aaa=np.array(apo_apo)
aba=np.array(apo_ben)
bba=np.array(ben_ben)

ax1.set_xlim(0.70,1.0)
#ax1.hist([apo_apo,apo_ben,ben_ben],bins=20,label=["apo-apo","apo-ben", "ben-ben"],sigma=0.5)
ax1.hist(aaa,bins=20,alpha=0.5,label="AA")
ax1.hist(aba,bins=20,alpha=0.5,label="AB")
ax1.hist(bba,bins=20,alpha=0.5,label="BB")
ax1.legend(loc="upper left")

outfile=open("results.dat","w")
outfile.write("AA(sigma,loc,scale)=%12.5f %12.5f %12.5f\n"% (sigma_aa, loc_aa, scale_aa))
outfile.write("AB(sigma,loc,scale)=%12.5f %12.5f %12.5f\n"% (sigma_ab, loc_ab, scale_ab))
outfile.write("BB(sigma,loc,scale)=%12.5f %12.5f %12.5f\n"% (sigma_bb, loc_bb, scale_bb))

outfile.write("AA(mean,std,median)=%12.5f %12.5f %12.5f\n"% (aaa.mean(), aaa.std(), np.median(aaa)))
outfile.write("AB(mean,std,median)=%12.5f %12.5f %12.5f\n"% (aba.mean(), aba.std(), np.median(aba)))
outfile.write("BB(mean,std,median)=%12.5f %12.5f %12.5f\n"% (bba.mean(), bba.std(), np.median(bba)))
outfile.close()

#plt.savefig("cc_dist.png")

Z = hierarchy.linkage(dis_list, 'ward')
title_result="\nAA(mean:%5.3f std:%5.3f median:%5.3f) AB(mean:%5.3f std:%5.3f median:%5.3f) BB(mean:%5.3f std:%5.3f median:%5.3f)" % \
    (aaa.mean(), aaa.std(), np.median(aaa), \
    aba.mean(), aba.std(), np.median(aba), \
    bba.mean(), bba.std(), np.median(bba))

print(title_result)
plt.title(title_s+title_result)

dn = hierarchy.dendrogram(Z,labels=sample_list, leaf_font_size=10)

# 最後のから２つ目、で、一番高い山のWard distanceを取得
last_merge = Z[-2]  
threshold = last_merge[2]  
print("Threshold for two main clusters:", threshold)
plt.annotate(f"Threshold: {threshold:.4f}", xy=(0.6, 0.65), xycoords='axes fraction')

plt.savefig("%s.jpg"%figname)
plt.show()
