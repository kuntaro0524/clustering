import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.stats import skewnorm
from scipy.cluster.hierarchy import fcluster

#scale=float(sys.argv[1])
n_total=int(sys.argv[1])
n_each = int(n_total/2.0)

# alpha, loc, scale
# This is 'apo-apo' optimistic data
#alpha = -13.851
#loc = 0.97
#scale = 0.0174

#alpha = -10.704
#loc = 0.97
#scale = 0.02343

# Benz-Benz
#alpha = -9.5818
#loc = 0.97
#scale = 0.0354

# Average
# alpha=-11.6908
# scale = 0.0241
# loc = 0.97

# Apo-Apo distribution -> best fitting
alpha=-15.2177
scale = 0.0195
loc = 0.97

# Differencial value 
delta = 0.01
cc_same = loc
cc_diff = loc - delta

sample_dict=[{"name":"A-A",  "alpha":alpha,"loc":cc_same,"scale":scale},
             {"name":"A-B", "alpha":alpha,"loc":cc_diff,"scale":scale},
             {"name":"B-B","alpha":alpha,"loc":cc_same,"scale":scale}]

# Figure name
figname="alpha_%.1f_%.3f_%.3f_N%05d" % (
    sample_dict[1]['alpha'],sample_dict[1]['loc'], sample_dict[1]['scale'], n_total)
    

files=glob.glob("%s*"%figname)
n_exist=len(files)
if n_exist != 0:
    index=n_exist
    figname="%s_%02d"%(figname,index)

# Dendrogram title
title_s = "INPUT: (%s: alpha: %8.3f loc:%8.3f scale:%8.3f)(%s alpha:%8.3f loc:%8.3f scale:%8.3f)(%s: alpha:%8.3f loc:%8.3f scale:%8.3f)" \
    % (sample_dict[0]['name'], sample_dict[0]['alpha'],sample_dict[0]['loc'], sample_dict[0]['scale'], \
    sample_dict[1]['name'], sample_dict[1]['alpha'],sample_dict[1]['loc'], sample_dict[1]['scale'], \
    sample_dict[2]['name'], sample_dict[2]['alpha'],sample_dict[2]['loc'], sample_dict[2]['scale'])

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
#ax1.hist([apo_apo,apo_ben,ben_ben],bins=20,label=["apo-apo","apo-ben", "ben-ben"],alpha=0.5)
ax1.hist(aaa,bins=20,alpha=0.5,label="AA")
ax1.hist(aba,bins=20,alpha=0.5,label="AB")
ax1.hist(bba,bins=20,alpha=0.5,label="BB")
ax1.legend(loc="upper left")

outfile=open("results.dat","w")
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

last_merge = Z[-2]  # 最後の結合を取得
threshold = last_merge[2]  # 最後の結合でのWard距離を取得
print("Threshold for two main clusters:", threshold)
print(last_merge)

plt.savefig("%s.jpg"%figname)
plt.show()