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

n_total=int(sys.argv[1])
n_each= int(n_total/2)

# クラスタリングの評価を行うクラス関数
# 入力は、クラスタリングで得られた結果のZと、クラスタリングの対象となったデータセットのラベルのリスト、しきい値である
# fcluster関数を利用して、クラスタリングがしきい値内でいくつのクラスタに分かれたかを計算する
# ここでは２種類に分類できたかどうかが重要であるため、しきい値以下のクラスタの数が２でなければFalseを返す
# ２つに分類できた場合は各クラスタに含まれるラベルのインデックスを cluster_1_indices, cluster_2_indicesに格納する
# さらに、クラスタリングの結果のラベルをcluster_1_label, cluster_2_labelに格納する
# 結果としてcluster 1と2に入っているラベルが"A", "B"である数を数える
# 最初に cluster 1に入っているデータの数をcluster_1_count, cluster 2に入っているデータの数をcluster_2_countとする
# cluster 1に入っているAの数が cluster_1_A_count, cluster 1に入っているBの数が cluster_1_B_count
# cluster 2に入っているAの数が cluster_2_A_count, cluster 1に入っているBの数が cluster_2_B_count
# とした場合、cluster 1と cluster 2 の"A", "B"の純度が計算できる (例は以下)
# cluster_1_A_purity = cluster_1_A_count / cluster_1_count
# 最終的に返り値は以下とする
# cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count

def evaluateClustering(Z, sample_list):
    from scipy.cluster.hierarchy import fcluster
    import numpy as np

    # fclusterを利用して２つのクラスタに分かれるしきい値を発見する
    thresh = 1.5
    while True:
        cluster_indices = fcluster(Z, thresh, criterion='distance')
        num_clusters = len(np.unique(cluster_indices))
        if num_clusters == 2:
            break
        if thresh < 0.1:
            # Exceptionをraiseする
            raise Exception("ERRORR")
            
        thresh -= 0.1
    
    if num_clusters != 2:
        # Exception をraiseする
        raise Exception("ERRORR")

    # print(cluster_indices)

    cluster_1_indices = np.where(cluster_indices == 1)[0]
    cluster_2_indices = np.where(cluster_indices == 2)[0]

    # print(cluster_1_indices)
    # print(cluster_2_indices)
    cluster_1_label = sample_list[cluster_1_indices]
    cluster_2_label = sample_list[cluster_2_indices]

    cluster_1_count = len(cluster_1_label)
    cluster_2_count = len(cluster_2_label)

    cluster_1_A_count = len(np.where(cluster_1_label == "A")[0])
    cluster_1_B_count = len(np.where(cluster_1_label == "B")[0])
    cluster_2_A_count = len(np.where(cluster_2_label == "A")[0])
    cluster_2_B_count = len(np.where(cluster_2_label == "B")[0])

    cluster_1_A_purity = cluster_1_A_count / cluster_1_count
    cluster_1_B_purity = cluster_1_B_count / cluster_1_count
    cluster_2_A_purity = cluster_2_A_count / cluster_2_count
    cluster_2_B_purity = cluster_2_B_count / cluster_2_count

    return cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count

def getScore(n1,n2, cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity):
    alpha = 0.6
    score_balance = 1 - abs(n1 - n2) / (n1 + n2)
    return score_balance

def proc(delta_loc, scale_scale,filename):
    #sigma_aa,loc_aa,scale_aa=0.40937419,-0.00289777, 0.0162868
    #sigma_bb,loc_bb,scale_bb=1.01742228,0.00229134,0.01365336
    #sigma_ab,loc_ab,scale_ab=0.26456247,-0.00657071, 0.03214396

    aaa=[0.7044,0.0027,0.0131]
    bba=[0.7648,0.0060,0.0247]
    aba=[0.7871,0.0147,0.0250]
    
    sigma_aa = aaa[0]
    loc_aa = aaa[1]
    scale_aa = aaa[2]
    
    sigma_bb = bba[0]
    loc_bb = bba[1]
    scale_bb = bba[2]
    
    sigma_ab = aba[0]
    loc_ab = aba[1]
    scale_ab = aba[2]

    # loc_bb = loc_bb - delta_loc
    loc_ab = loc_ab - delta_loc

    # scale scale
    scale_aa = scale_aa * scale_scale
    scale_bb = scale_bb * scale_scale
    scale_ab = scale_ab * scale_scale

    sample_dict=[{"name":"A-A","sigma":sigma_aa,"loc":loc_aa,"scale":scale_aa},
             {"name":"A-B","sigma":sigma_ab,"loc":loc_ab,"scale":scale_ab},
             {"name":"B-B","sigma":sigma_bb,"loc":loc_bb,"scale":scale_bb}]


    try:
        Z,cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count= corePart(sample_dict, delta_loc, scale_scale, n_each, filename)
        print("######################################################")
        print(cluster_1_A_purity, cluster_1_B_purity)
        print("######################################################")
        score = getScore(cluster_1_count, cluster_2_count, cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity)
        print("### score = ", score)
    except Exception as e:
        print(e)
        print("Error in evaluateClustering")
        cluster_1_A_purity = 0.0
        cluster_1_B_purity = 0.0
        cluster_2_A_purity = 0.0
        cluster_2_B_purity = 0.0
        cluster_1_count = 0
        cluster_2_count = 0
        score=0.0

    # 最後から１つ目で、一番高い山のWard distanceを取得
    last_merge = Z[-1]
    thresh0 = last_merge[2]

    # 最後のから２つ目、で、一番高い山のWard distanceを取得
    last_merge = Z[-2]  
    threshold = last_merge[2]  

    # 新しい threshold
    new_thresh = threshold/thresh0

    #print("Threshold for two main clusters:", threshold)
    plt.annotate(f"Threshold: {threshold:.4f}", xy=(0.6, 0.65), xycoords='axes fraction')

    # new_threshもグラフ中に書き出す
    plt.annotate(f"New threshold: {new_thresh:.4f}", xy=(0.6, 0.6), xycoords='axes fraction')

    # 結果をファイルに書き出す
    # "nds.csv"がすでにあるかどうか
    import os
    if not os.path.exists("nds.csv"):
        outfile=open("nds.csv","w")
        # ヘッダーを書き出す
        outfile.write("nds,delta_loc,scale_scale,threshold,new_threshold,filename,cluster_1_A_purity,cluster_1_B_purity,cluster_2_A_purity, cluster_2_B_purity, cluster_1_count,cluster_2_count,score\n")
    else:
        outfile=open("nds.csv","a")
        # 書き出すのは
        # nds,delta_loc,scale_scale,threshold,new_threshold,filename,cluster_1_A_purity,cluster_1_B_purity,cluster_2_A_purity, cluster_2_B_purity, cluster_1_count,cluster_2_count,score
        outfile.write("%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%s,%8.4f,%8.4f,%8.4f,%8.4f,%d,%d,%8.4f\n" % (n_total, delta_loc, scale_scale, threshold, new_thresh, filename, cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count, score))
    plt.savefig(filename)

def corePart(sample_dict, delta_loc, scale_scale, n_each, filename):
    # Dendrogram title
    title="delta_loc:%8.4f scale_scale:%8.4f" % (delta_loc, scale_scale)
        
    def get_stat_info(cc_combination):
        for idx,s in enumerate(sample_dict):
            if s['name']==cc_combination:
                return s

    sample_list=[]
    for i in np.arange(0,n_each):
        sample_list.append("A")
    for i in np.arange(0,n_each):
        sample_list.append("B")

    sample_list = np.array(sample_list)

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
    ax1.hist(aaa,bins=50,alpha=0.5,label="AA")
    ax1.hist(aba,bins=50,alpha=0.5,label="AB")
    ax1.hist(bba,bins=50,alpha=0.5,label="BB")
    ax1.legend(loc="upper left")

    Z = hierarchy.linkage(dis_list, 'ward')
    title_result="\nAA(mean:%5.3f std:%5.3f median:%5.3f) AB(mean:%5.3f std:%5.3f median:%5.3f) BB(mean:%5.3f std:%5.3f median:%5.3f)" % \
        (aaa.mean(), aaa.std(), np.median(aaa), \
        aba.mean(), aba.std(), np.median(aba), \
        bba.mean(), bba.std(), np.median(bba))

    #print(title_result)
    plt.title(title+title_result)

    dn = hierarchy.dendrogram(Z,labels=sample_list, leaf_font_size=10)

    # evaluate the threshold
    # 例外処理を入れつつ計算をする
    try:
        cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count= evaluateClustering(Z, sample_list)
    except Exception as e:
        print(e)
        print("Error in evaluateClustering")
        cluster_1_A_purity = 0.0
        cluster_1_B_purity = 0.0
        cluster_2_A_purity = 0.0
        cluster_2_B_purity = 0.0
        cluster_1_count = 0
        cluster_2_count = 0

    # 最後から１つ目で、一番高い山のWard distanceを取得
    last_merge = Z[-1]
    thresh0 = last_merge[2]

    # 最後のから２つ目、で、一番高い山のWard distanceを取得
    last_merge = Z[-2]  
    threshold = last_merge[2]  

    # 新しい threshold
    new_thresh = threshold/thresh0

    print("# of datasets:", len(sample_list))
    print("Threshold for two main clusters:", threshold)
    print("New threshold:", new_thresh)
    plt.annotate(f"Threshold: {threshold:.4f}", xy=(0.6, 0.65), xycoords='axes fraction')

    # new_threshもグラフ中に書き出す
    plt.annotate(f"New threshold: {new_thresh:.4f}", xy=(0.6, 0.6), xycoords='axes fraction') 
    # cluster_1_count, cluster_2_countもグラフ中に書き出す
    plt.annotate(f"Cluster 1 count: {cluster_1_count}", xy=(0.6, 0.55), xycoords='axes fraction')
    plt.annotate(f"Cluster 2 count: {cluster_2_count}", xy=(0.6, 0.5), xycoords='axes fraction')
    # cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purityもグラフ中に書き出す
    plt.annotate(f"Cluster 1 A purity: {cluster_1_A_purity:.4f}", xy=(0.6, 0.45), xycoords='axes fraction')
    plt.annotate(f"Cluster 1 B purity: {cluster_1_B_purity:.4f}", xy=(0.6, 0.4), xycoords='axes fraction')
    plt.annotate(f"Cluster 2 A purity: {cluster_2_A_purity:.4f}", xy=(0.6, 0.35), xycoords='axes fraction')
    plt.annotate(f"Cluster 2 B purity: {cluster_2_B_purity:.4f}", xy=(0.6, 0.3), xycoords='axes fraction')

    plt.savefig(filename)

    return Z,cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count

#bba=[0.7648,0.0060,0.0247]
#aba=[0.7871,0.0147,0.0250]

diff = 0.0147 - 0.0060

# 各パラメータで10回ずつ計算する
for i in range(100):
    # diff を 10で割ってそれぞれの数値をdeltaに入れる   
    for j in range(10):
        delta = 0.0060 + diff/10*j
        filename = "delta_%08.5f_%03d.png" % (delta,i)
        proc(delta,1.0,filename)
