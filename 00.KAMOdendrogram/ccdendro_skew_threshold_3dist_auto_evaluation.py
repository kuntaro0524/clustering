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
# AA
alpha_aa = -15.2177
loc_aa = 0.9945
scale_aa = 0.0195

# BB
alpha_bb = -8.4750
loc_bb = 0.9883
scale_bb = 0.0251

# AB
alpha_ab = -11.3798
loc_ab = 0.9799
scale_ab = 0.0277

from scipy.optimize import minimize_scalar

def run(stat_dict,outfile,idx):
    # FigureのPrefixを作成
    # loc_abの数値を入れたものを作成
    figname = "loc_ab_%.4f_%02d" % (stat_dict[2]['loc'],idx)

    # Dendrogram title
    title_s = "INPUT: (%s: alpha: %8.3f loc:%8.3f scale:%8.3f)(%s alpha:%8.3f loc:%8.3f scale:%8.3f)(%s: alpha:%8.3f loc:%8.3f scale:%8.3f)" \
        % (sample_dict[0]['name'], sample_dict[0]['alpha'],sample_dict[0]['loc'], sample_dict[0]['scale'], \
        sample_dict[1]['name'], sample_dict[1]['alpha'],sample_dict[1]['loc'], sample_dict[1]['scale'], \
        sample_dict[2]['name'], sample_dict[2]['alpha'],sample_dict[2]['loc'], sample_dict[2]['scale'])

    # Fittingによるピーク座標の取得関数
    def get_peak_code(stat_dict):
        peak_codes=[]
        for idx,each_cond in enumerate(sample_dict):
            alpha, loc, scale = each_cond['alpha'], each_cond['loc'], each_cond['scale']

            # 最小化の対象となる関数（pdf関数を負にした関数）
            def neg_skewed_gaussian(x, alpha, loc, scale):
                return -skewnorm.pdf(x, alpha, loc, scale)

            # 最小化問題を解く
            result = minimize_scalar(neg_skewed_gaussian, args=(alpha, loc, scale))
            # X 範囲
            xdata = np.arange(0,1.1,0.001)
            ydata=-neg_skewed_gaussian(xdata,alpha,loc,scale)

            # ピーク座標（Mode）を取得
            mode_x = result.x
            mode_y = skewnorm.pdf(mode_x, alpha, loc, scale)
            print(f"Mode position: x = {mode_x}, y = {mode_y}")

            peak_codes.append(mode_x)

        return peak_codes

    peak_codes = get_peak_code(stat_dict)

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

    ofile=open("cc_%02d.dat" % idx,"w")

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

    # 最瀕値を取得する関数を定義する
    def find_mode(data):
        # ヒストグラムのbinの数を計算
        num_bins = len(data) // 10

        # ヒストグラムを作成
        hist, bin_edges = np.histogram(data, bins=num_bins)

        # 最も頻度が高いbinのインデックスを取得
        max_bin_index = np.argmax(hist)

        # 最瀕値の範囲の最小値と最大値を取得
        min_val = bin_edges[max_bin_index]
        max_val = bin_edges[max_bin_index + 1]

        # 最瀕値の最終値（最小値と最大値の平均）を計算
        mode_final = (min_val + max_val) / 2

        return mode_final

    # クラスタリングと評価関数
    # Z: クラスタリング結果
    def evaluate_clustering(Z, labels, threshold=None):
        if threshold is None:
            threshold = Z[-1, 2] / 2  # デフォルトのしきい値を設定

        clusters = fcluster(Z, threshold, criterion='distance')
        num_clusters = len(np.unique(clusters))

        if num_clusters != 2:
            return False, "Incorrect number of clusters: {}".format(num_clusters)

        # ２つに分かれたと仮定して、それぞれのクラスタに含まれるラベルのインデックスを取得
        cluster_1_indices = np.where(clusters == 1)[0]
        cluster_2_indices = np.where(clusters == 2)[0]

        # それぞれのクラスタに含まれるラベルを取得
        cluster_1_labels = labels[cluster_1_indices]
        cluster_2_labels = labels[cluster_2_indices]

        #print(cluster_1_labels)
        #print(cluster_2_labels)

        # Cluster 1 & 2 にはいっているA,Bの数をカウントする
        cluster_1_A_count = np.sum(cluster_1_labels == "A")
        cluster_1_B_count = np.sum(cluster_1_labels == "B")
        cluster_2_A_count = np.sum(cluster_2_labels == "A")
        cluster_2_B_count = np.sum(cluster_2_labels == "B")

        # それぞれのクラスタに含まれるラベルの数
        n_1_label = len(cluster_1_labels)
        n_2_label = len(cluster_2_labels)
        print(f"# of data in cluster1 {n_1_label:10d}")
        print(f"# of data in cluster2 {n_2_label:10d}")
        print(f"# of A in cluster1 {cluster_1_A_count:10d}")
        print(f"# of B in cluster1 {cluster_1_B_count:10d}")
        print(f"# of A in cluster2 {cluster_2_A_count:10d}")
        print(f"# of B in cluster2 {cluster_2_B_count:10d}")

        # それぞれのクラスタに含まれるラベルの純度を計算
        cluster_1_A_purity = cluster_1_A_count / n_1_label
        cluster_1_B_purity = cluster_1_B_count / n_1_label
        cluster_2_A_purity = cluster_2_A_count / n_2_label
        cluster_2_B_purity = cluster_2_B_count / n_2_label

        return True, cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, n_1_label, n_2_label, peak_codes[0], peak_codes[1], peak_codes[2]

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

    # それぞれの分布の最頻値を求める
    mode_aa = find_mode(aaa)
    mode_ab = find_mode(aba)
    mode_bb = find_mode(bba)

    ax1.set_xlim(0.85,1.0)
    ax1.hist(aaa,bins=20,alpha=0.5,label="AA", density=True)
    ax1.hist(aba,bins=20,alpha=0.5,label="AB", density=True)
    ax1.hist(bba,bins=20,alpha=0.5,label="BB",density=True)
    ax1.legend(loc="upper left")

    Z = hierarchy.linkage(dis_list, 'ward')
    title_result="\nAA(mean:%5.3f std:%5.3f median:%5.3f) AB(mean:%5.3f std:%5.3f median:%5.3f) BB(mean:%5.3f std:%5.3f median:%5.3f)" % \
        (aaa.mean(), aaa.std(), np.median(aaa), \
        aba.mean(), aba.std(), np.median(aba), \
        bba.mean(), bba.std(), np.median(bba))

    plt.title(title_s+title_result)

    dn = hierarchy.dendrogram(Z,labels=sample_list, leaf_font_size=10)

    results_array=evaluate_clustering(Z, labels=np.array(sample_list), threshold=0.6)

    if results_array[0] == False:
        comment = results_array[1]
        cluster1_purity_A=-999.999
        cluster1_purity_B=-999.999
        cluster2_purity_A=-999.999
        cluster2_purity_B=-999.999
        n_1_label=0
        n_2_label=0
        peak_AA=0
        peak_BB=0
        peak_AB=0
    else:
        cluster1_purity_A=results_array[1]
        cluster1_purity_B=results_array[2]
        cluster2_purity_A=results_array[3]
        cluster2_purity_B=results_array[4]
        n_1_label=results_array[5]
        n_2_label=results_array[6]
        peak_AA=results_array[7]
        peak_BB=results_array[8]
        peak_AB=results_array[9]

        # Real evaluation
        if (cluster1_purity_A > 0.8 and cluster2_purity_B > 0.8) or \
            (cluster1_purity_B > 0.8 and cluster2_purity_A > 0.8):
            comment="Classificated."
        else:
            comment="Not classificated."

    last_merge = Z[-2]  # 最後の結合を取得
    threshold = last_merge[2]  # 最後の結合でのWard距離を取得

    plt.annotate(f"mode(AA)= {mode_aa:.3f}", xy=(0.5, 0.85), xycoords='axes fraction')
    plt.annotate(f"mode(BB)= {mode_bb:.3f}", xy=(0.5, 0.80), xycoords='axes fraction')
    plt.annotate(f"mode(AB)= {mode_ab:.3f}", xy=(0.5, 0.75), xycoords='axes fraction')
    plt.annotate(f"Threshold for two main clusters: {threshold}", xy=(0.5, 0.70), xycoords='axes fraction')

    plt.savefig("%s.jpg"%figname)
    #plt.show()

    # Delta values
    delta_aa_ab = mode_aa - mode_ab
    delta_bb_ab = mode_bb - mode_ab

    outfile.write("%8.5f,%8.5f,%8.5f,"% (loc_aa, loc_bb, loc_ab))
    outfile.write("%8.5f,%8.5f,%8.5f,"% (aaa.mean(), aaa.std(), np.median(aaa)))
    outfile.write("%8.5f,%8.5f,%8.5f,"% (bba.mean(), bba.std(), np.median(bba)))
    outfile.write("%8.5f,%8.5f,%8.5f,"% (aba.mean(), aba.std(), np.median(aba)))
    outfile.write("%8.5f,%8.5f,%8.5f," % (mode_aa, mode_bb, mode_ab))
    outfile.write("%8.5f,%8.5f," % (delta_aa_ab, delta_bb_ab))
    outfile.write("%8.5f,%8.5f,%8.5f,%8.5f," % (cluster1_purity_A, cluster1_purity_B, cluster2_purity_A, cluster2_purity_B))
    outfile.write("%5d,%5d," % (n_1_label, n_2_label))
    outfile.write("%8.5f,%8.5f,%8.5f," % (peak_AA, peak_BB, peak_AB))
    outfile.write("%s\n" % (comment))

# alpha_ab = -11.3798
# loc_ab = 0.988
# scale_ab = 0.0277

outfile=open("result.dat","w")
outfile.write("loc_AA,loc_BB,loc_AB,mean_AA,std_AA,med_AA,mean_BB,std_BB,med_BB,mean_AB,std_AB,med_AB,mode_AA,mode_BB,mode_AB,delta_AA_AB,delta_BB_AB,purity_1_A,purity_1_B,purity_2_A,purity_2_B,n1,n2,peak_AA,peak_BB,peak_AB,comment\n")

# Changing value
#for loc_ab in np.arange(0.9699, 0.9999, 0.001):
for loc_ab in np.arange(0.985, 0.989, 0.0005):
    for idx in np.arange(0,10):
        print("IDX=%d" % idx)
        sample_dict=[{"name":"A-A","alpha":alpha_aa,"loc":loc_aa,"scale":scale_aa},
                 {"name":"B-B","alpha":alpha_bb,"loc":loc_bb,"scale":scale_bb},
                 {"name":"A-B","alpha":alpha_ab,"loc":loc_ab,"scale":scale_ab}]
        run(sample_dict, outfile, idx)