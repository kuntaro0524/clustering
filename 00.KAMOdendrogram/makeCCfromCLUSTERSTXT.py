import sys
import pandas as pd
import numpy as np
# scipy curve_fit をインポート
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
from matplotlib import pyplot as plt

def get_id_list_from_clusters(cluster_number, file_path="CLUSTERS.txt"):
    id_list = []

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            cols = line.split()
            print(cols[0])
            if cols[0] == cluster_number:
                id_list = [int(ID) - 1 for ID in cols[3:]]
                break

    return id_list

# ２つのクラスタIDを受け取り、CLUSTERS.txtからそのクラスタに含まれるID番号のリストを取得
# その際、cctable.datからCCの値を取得する
# 抜き出す条件は、cctable.datのi,jの組のうち、「iが指定した最初のクラスタIDに所属している」かつ「jが指定した2番目のクラスタIDに所属している」
# または　「iが指定した2番目のクラスタIDに所属している」かつ「jが指定した最初のクラスタIDに所属している」
# とする

def get_cc_cross_values_from_cctable(cluster_number1, cluster_number2, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst"):
    id_list1 = get_id_list_from_clusters(cluster_number1, clusters_file)
    id_list2 = get_id_list_from_clusters(cluster_number2, clusters_file)

    cc_values = []

    # filename_listを読み込む(リストに格納)
    filename_list = []
    with open(listname, "r") as f:
        lines = f.readlines()
        for line in lines:
            filename_list.append(line.strip())

    cctype_list=[]

    with open(cctable_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            cols = line.split()
            i, j = int(cols[0]), int(cols[1])

            if (i in id_list1 and j in id_list2) or (i in id_list2 and j in id_list1):
                name_i = filename_list[i]
                name_j = filename_list[j]
                cctype_list.append(name_i + "_" + name_j)
                cc_values.append(float(cols[2]))

    # cc_values が格納されたDataFrameを返す
    ret = pd.DataFrame(cc_values, columns=["cc"])
    # retにcctype_listを追加する
    ret["cctype"] = cctype_list
    
    return ret

def get_cc_values_from_cctable(cluster_number, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst"):
    id_list = get_id_list_from_clusters(cluster_number, clusters_file)

    cc_values = []

    # filename_listを読み込む(リストに格納)
    filename_list = []
    with open(listname, "r") as f:
        lines = f.readlines()
        for line in lines:
            filename_list.append(line.strip())

    cctype_list=[]

    with open(cctable_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            cols = line.split()
            i, j = int(cols[0]), int(cols[1])

            if i in id_list and j in id_list:
                name_i = filename_list[i]
                name_j = filename_list[j]
                cctype_list.append(name_i + "_" + name_j)
                cc_values.append(float(cols[2]))

    # cc_values が格納されたDataFrameを返す
    ret = pd.DataFrame(cc_values, columns=["cc"])
    # retにcctype_listを追加する
    ret["cctype"] = cctype_list
    
    return ret

# CCのDataFrameを受け取って、CCの分布をヒストグラムをskeewed gaussianでフィッティングする
def fit(cc_df, cc_threshold = 0.8, nbins=20):
    # CCの数値が0.8以上のものだけに限定する
    filter_condition = cc_df['cc'] >= cc_threshold
    cc_df = cc_df[filter_condition]
    # CCの標準偏差を計算する
    sigma_data = np.std(cc_df['cc'])
    # CCの標準偏差を表示する
    print("sigma_data: ", sigma_data)
    # CC data array
    ccdata = cc_df['cc']

    # cc type list
    cctype_list = cc_df['cctype']

    # 初期値？
    hist, bin_edges = np.histogram(ccdata, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # skewed gaussian 関数の定義
    def skewed_gaussian(x, alpha, loc, scale):
        rtn_value = skewnorm.pdf(x, alpha, loc, scale)
        return rtn_value

    # 初期値の設定と最小二乗法の計算
    initial_guess = [0, np.mean(ccdata), np.std(ccdata)]
    popt, pcov = curve_fit(skewed_gaussian, bin_centers, hist, p0=initial_guess)
    alpha_fit, loc_fit, scale_fit = popt

    # Histogram drawing
    plt.hist(ccdata, bins=nbins, color='green', alpha=0.5, density=True, label='CC Histogram')
    plt.xlabel('CC')
    plt.ylabel('Frequency')

    # Plotting the resultant skewed gaussian function
    plt.plot(bin_centers, skewed_gaussian(bin_centers, *popt), 'r-', label='Fitted Skew Gaussian')

    plt.annotate(f"Alpha: {alpha_fit:.4f}", xy=(0.6, 0.85), xycoords='axes fraction')
    plt.annotate(f"Loc: {loc_fit:.4f}", xy=(0.6, 0.75), xycoords='axes fraction')
    plt.annotate(f"Scale: {scale_fit:.4f}", xy=(0.6, 0.65), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <cluster_number1> <cluster_number2> ...")
        sys.exit(1)

    cluster_numbers = sys.argv[1:]
    cc_values_list = []
    ncluster=int(sys.argv[3])

    #df1 = get_cc_values_from_cctable(cluster_number[0])
    #df2 = get_cc_values_from_cctable(cluster_number[1])
    df3 = get_cc_cross_values_from_cctable(cluster_numbers[0], cluster_numbers[1])

    #print(f"CC values for cluster {cluster_number}: {cc_values}")
    fit(cc_df=df3, cc_threshold = 0.8, nbins=ncluster)
    #cc_data = pd.DataFrame({'Cluster_Number': cluster_numbers, 'CC_Values': cc_values_list},ncluster)
    print(cc_data)