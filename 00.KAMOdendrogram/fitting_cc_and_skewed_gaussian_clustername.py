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
            #print(cols[0])
            if cols[0] == cluster_number:
                id_list = [int(ID) - 1 for ID in cols[3:]]
                break

    return id_list

def get_cc_various_values_from_cctable(cluster_number1, cluster_number2, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst"):
    id_list1 = get_id_list_from_clusters(cluster_number1, clusters_file)
    id_list2 = get_id_list_from_clusters(cluster_number2, clusters_file)

    print(id_list1)
    print(id_list2)

    cc_values = []
    cctype_list=[]

    # 同じクラスタIDに含まれるインデックスどうし、または異なるクラスタIDに含まれるインデックスどうしのCC値を取得する
    # iとjの両方がid_list1に含まれている場合には対応するCC値を取得→type: "AA"をキーとしてccをdictに格納→配列として保存する
    # iがid_list1に含まれているか、jがid_list2に含まれている場合 またはiがid_list2に含まれているか、jがid_list1に含まれている場合には "AB" をキーとしてccをdictに格納→配列として保存する
    # iとjの両方がid_list2に含まれている場合には "BB" をキーとしてccをdictに格納→配列として保存する

    with open(cctable_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            cols = line.split()
            i, j = int(cols[0]), int(cols[1])
            # iとjの両方がid_list1に含まれている場合には対応するCC値を取得→type: "AA"をキーとしてccをdictに格納→配列として保存する
            if i in id_list1 and j in id_list1:
                cctype_list.append("AA")
                cc_values.append(float(cols[2]))
            # iとjの両方がid_list2に含まれている場合には "BB" をキーとしてccをdictに格納→配列として保存する
            elif i in id_list2 and j in id_list2:
                cctype_list.append("BB")
                cc_values.append(float(cols[2]))
            # それ以外の場合はABとなる
            else:
                cctype_list.append("AB")
                cc_values.append(float(cols[2]))

    # cc_values が格納されたDataFrameを返す
    ret = pd.DataFrame(cc_values, columns=["cc"])
    # retにcctype_listを追加する
    ret["cctype"] = cctype_list
    
    return ret

# CCのDataFrameを受け取って、CCの分布をヒストグラムをskeewed gaussianでフィッティングする
def fit(cc_df, cc_threshold = 0.8, figname="fig.png"):
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
    n_bins = int(len(ccdata) / 8.0)
    hist, bin_edges = np.histogram(ccdata, bins=n_bins)
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
    plt.hist(ccdata, bins=n_bins, color='green', alpha=0.5, label='CC Histogram')
    plt.xlabel('CC')
    plt.ylabel('Frequency')

    # Plotting the resultant skewed gaussian function
    plt.plot(bin_centers, skewed_gaussian(bin_centers, *popt), 'r-', label='Fitted Skew Gaussian')

    plt.annotate(f"Alpha: {alpha_fit:.4f}", xy=(0.6, 0.85), xycoords='axes fraction')
    plt.annotate(f"Loc: {loc_fit:.4f}", xy=(0.6, 0.75), xycoords='axes fraction')
    plt.annotate(f"Scale: {scale_fit:.4f}", xy=(0.6, 0.65), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(figname)
    plt.clf()

    return alpha_fit, loc_fit, scale_fit

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <cluster_number1> <cluster_number2> ...")
        sys.exit(1)

    cluster_numbers = sys.argv[1:]
    cc_values_list = []

    ret=get_cc_various_values_from_cctable(cluster_numbers[0], cluster_numbers[1], clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst")
    print(ret)

    # 'cc'の数値が0.8以上のものだけ抜く
    cond = ret['cc'] >=0.8
    df_filter = ret[cond]

    # cctypeが"AA"のものだけを抽出する
    df1 = df_filter[df_filter['cctype'] == "AA"]
    plt.hist(df1['cc'],alpha=0.5,density=True)
    plt.savefig("1.png")
    plt.clf()
    print(f"AA: {len(df1)}")
    # cctypeが"BB"のものだけを抽出する
    df2 = df_filter[df_filter['cctype'] == "BB"]
    plt.hist(df2['cc'],alpha=0.5,density=True)
    plt.savefig("2.png")
    plt.clf()
    print(f"BB: {len(df2)}")
    # cctypeが"AB"のものだけを抽出する
    df3 = df_filter[df_filter['cctype'] == "AB"]
    plt.hist(df3['cc'],alpha=0.5,density=True)
    plt.savefig("3.png")
    plt.clf()
    print(f"AB: {len(df3)}")

    # matplotlib で水平方向に３みつのグラフを作成
    #fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    alpha1,loc1,scale1=fit(cc_df=df1, cc_threshold = 0.8, figname="aa_fit.png")
    alpha2,loc2,scale2=fit(cc_df=df2, cc_threshold = 0.8, figname="bb_fit.png")
    alpha3,loc3,scale3=fit(cc_df=df3, cc_threshold = 0.8, figname="ab_fit.png")

    # 結果を表示 alpha1, loc1, scale1を1行で表示
    print(f"Cluster {cluster_numbers[0]}: alpha={alpha1:.4f}, loc={loc1:.4f}, scale={scale1:.4f}")
    print(f"Cluster {cluster_numbers[1]}: alpha={alpha2:.4f}, loc={loc2:.4f}, scale={scale2:.4f}")
    print(f"Cluster {cluster_numbers[0]} and {cluster_numbers[1]}: alpha={alpha3:.4f}, loc={loc3:.4f}, scale={scale3:.4f}")