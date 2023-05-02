import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.stats import skewnorm
from scipy.cluster.hierarchy import fcluster
from scipy.stats import lognorm



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

n_each=50

def proc(delta_loc, scale_scale,filename):
    sigma_aa,loc_aa,scale_aa=0.40937419,-0.00289777, 0.0162868
    sigma_bb,loc_bb,scale_bb=1.01742228,0.00229134,0.01365336
    sigma_ab,loc_ab,scale_ab=0.26456247,-0.00657071, 0.03214396

    sample_dict=[{"name":"A-A","sigma":sigma_aa,"loc":loc_aa,"scale":scale_aa},
             {"name":"A-B","sigma":sigma_ab,"loc":loc_ab,"scale":scale_ab},
             {"name":"B-B","sigma":sigma_bb,"loc":loc_bb,"scale":scale_bb}]

    loc_bb = loc_bb - delta_loc
    loc_ab = loc_ab - delta_loc

    # scale scale
    scale_aa = scale_aa * scale_scale
    scale_bb = scale_bb * scale_scale
    scale_ab = scale_ab * scale_scale

    # Dendrogram title
    title="delta_loc:%8.3f scale_scale:%8.3f" % (delta_loc, scale_scale)
        
    def get_stat_info(cc_combination):
        for idx,s in enumerate(sample_dict):
            if s['name']==cc_combination:
                return s

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

    Z = hierarchy.linkage(dis_list, 'ward')
    title_result="\nAA(mean:%5.3f std:%5.3f median:%5.3f) AB(mean:%5.3f std:%5.3f median:%5.3f) BB(mean:%5.3f std:%5.3f median:%5.3f)" % \
        (aaa.mean(), aaa.std(), np.median(aaa), \
        aba.mean(), aba.std(), np.median(aba), \
        bba.mean(), bba.std(), np.median(bba))

    print(title_result)
    plt.title(title+title_result)

    dn = hierarchy.dendrogram(Z,labels=sample_list, leaf_font_size=10)

    # 最後から１つ目で、一番高い山のWard distanceを取得
    last_merge = Z[-1]
    thresh0 = last_merge[2]

    # 最後のから２つ目、で、一番高い山のWard distanceを取得
    last_merge = Z[-2]  
    threshold = last_merge[2]  

    # 新しい threshold
    new_threshodl = threshold/thresh0

    print("Threshold for two main clusters:", threshold)
    plt.annotate(f"Threshold: {threshold:.4f}", xy=(0.6, 0.65), xycoords='axes fraction')

    # 結果をファイルに書き出す
    outfile=open("nds.csv","a")
    # delta_loc, scale_scale, threshold, new_threshold
    outfile.write("%8.3f,%8.3f,%8.3f,%8.3f\n" % (delta_loc, scale_scale, threshold, new_threshodl))
    plt.savefig(filename)


for delta in np.arange(0,0.1,0.01):
    for scale in np.arange(1.0, 5.0,0.1):
        filename = "delta_%08.5f_scale_%08.5f.png" % (delta, scale)
        proc(delta,scale,filename)