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

# [0.57589715 0.0033695  0.02383734]
# [0.93177759 0.00324547 0.01636987]
# [0.7316575  0.01419515 0.02412596]

# 230418 Yabe
# [0.60780605 0.00202836 0.01278435]
# [0.89524981 0.01490806 0.00835992]
# [0.43960031 0.01302677 0.0185993 ]

# 230418 CC>= 0.8
#[0.60780605 0.00202836 0.01278435]
#[0.89524981 0.01490806 0.00835992]
#[0.43960031 0.01302677 0.0185993 ]

# 230421
# [('AA', array([0.72011622, 0.00189438, 0.01493311])), ('BB', array([0.53182721, 0.01370284, 0.01926178])), ('AB', array([0.91875735, 0.00667211, 0.02909116]))]

sigma_aa = 0.720
loc_aa = 0.00189
scale_aa = 0.0149

sigma_bb= 0.5318
loc_bb= 0.0137
scale_bb= 0.019

sigma_ab=0.9187
loc_ab=0.0066
scale_ab=0.0290

# 正直にめっちゃあうようにしたやつ
# 100 sets で threshold = 0.65
sigma_aa = 0.7044
loc_aa = 0.0027
scale_aa = 0.0131

sigma_bb = 0.7648
loc_bb = 0.0060
scale_bb = 0.0247

sigma_ab = 0.7871
loc_ab = 0.0147
scale_ab = 0.0250

sample_dict=[{"name":"apo-apo","sigma":sigma_aa,"loc":loc_aa,"scale":scale_aa},
             {"name":"apo-benz","sigma":sigma_ab,"loc":loc_ab,"scale":scale_ab},
             {"name":"benz-benz","sigma":sigma_bb,"loc":loc_bb,"scale":scale_bb}]

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
    sample_list.append("apo")
for i in np.arange(0,n_each):
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
            name_list.append("A-A")
            cctmp = calcCCvalue(stat_dict)
            apo_apo.append(cctmp)
        elif s1=="benz" and s2=="benz":
            stat_dict=get_stat_info("benz-benz")
            name_list.append("benz-benz")
            cctmp = calcCCvalue(stat_dict)
            ben_ben.append(cctmp)
        else:
            stat_dict=get_stat_info("apo-benz")
            name_list.append("apo-benz")
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
fig = plt.figure(figsize=(15,9))
fig.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95) #この1行を入れる
# 上下左右の余白を広めにとる
spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 5])
#ax1=fig.add_subplot(spec[0])
#ax2=fig.add_subplot(spec[1])

# CC stats
aaa=np.array(apo_apo)
aba=np.array(apo_ben)
bba=np.array(ben_ben)
#ax1.set_xlim(0.90,1.0)
#ax1.hist([apo_apo,apo_ben,ben_ben],bins=20,label=["apo-apo","apo-ben", "ben-ben"],sigma=0.5)
#ax1.hist(aaa,bins=20,alpha=0.5,label="apo-apo")
#ax1.hist(aba,bins=20,alpha=0.5,label="apo-benz")
#ax1.hist(bba,bins=20,alpha=0.5,label="benz-benz")

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
#ax1.plot(x, yaa, label="AA")
#ax1.plot(x, ybb, label="BB")
#ax1.plot(x, yab, label="AB")
#ax1.legend(loc="upper left")

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
#title_result="\nAA(mean:%5.3f std:%5.3f median:%5.3f) AB(mean:%5.3f std:%5.3f median:%5.3f) BB(mean:%5.3f std:%5.3f median:%5.3f)" % \
    #(aaa.mean(), aaa.std(), np.median(aaa), \
    #aba.mean(), aba.std(), np.median(aba), \
    #bba.mean(), bba.std(), np.median(bba))

#plt.title(title_s+title_result)

dn = hierarchy.dendrogram(Z,labels=sample_list, leaf_font_size=10)

# 最後のから２つ目、で、一番高い山のWard distanceを取得
last_merge = Z[-2]  
threshold = last_merge[2]  
print("Threshold for two main clusters:", threshold)
# 横軸の名前は "individual datasets"とする
# xlabelの文字を20ptsにする
plt.xlabel("individual datasets", fontsize=30)
#　縦軸の名前を "Ward distance"とする
plt.ylabel("Ward distance", fontsize=30)
# yticsのフォントサイズを20ptsにする
plt.yticks(fontsize=20)
plt.xticks(fontsize=9)
plt.annotate(f"threshold={threshold:.3f}", xy=(0.4, 0.55), xycoords='axes fraction', fontsize=20)
plt.savefig("%s.png"%figname)
plt.show()
