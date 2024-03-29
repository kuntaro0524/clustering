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

# scale
scale_fac = float(sys.argv[2])

# 230421
sigma_aa,loc_aa,scale_aa=0.40937419,-0.00289777, 0.0162868
sigma_bb,loc_bb,scale_bb=1.01742228,0.00229134,0.01365336
sigma_ab,loc_ab,scale_ab=0.26456247,-0.00657071, 0.03214396

scale_aa = scale_aa * scale_fac
scale_bb = scale_bb * scale_fac
scale_ab = scale_ab * scale_fac

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

# 最後のクラスターを取得
last = Z[-1]
# 最後のクラスターの距離を取得
last_dist = last[2]

# 最後のから２つ目、で、一番高い山のWard distanceを取得
last_merge = Z[-2]  
threshold = last_merge[2]  
print("Threshold for two main clusters:", threshold)

# New ratio
new_threshold = threshold / last_dist
print("New threshold for two main clusters:", new_threshold)
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
