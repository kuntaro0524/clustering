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
    n_bins = int(len(ccdata) / 8)
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
    if len(sys.argv) != 3:
        print("Usage: python script.py <cluster_number1> <cluster_number2>")
        sys.exit(1)

    cluster_numbers = sys.argv[1:]
    cc_values_list = []

    ret=get_cc_various_values_from_cctable(cluster_numbers[0], cluster_numbers[1], clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst")
    print(ret)

    # cc threshold
    cc_thresh = 0.6
    # 'cc'の数値が0.8以上のものだけ抜く
    cond = ret['cc'] >=cc_thresh
    df_filter = ret[cond]

    # results は dictの配列
    results = []

    # cctypeが"AA"のものだけを抽出する
    df1 = df_filter[df_filter['cctype'] == "AA"]
    nAA = len(df1)
    # 例外を補足する
    try:
        alpha1,loc1,scale1=fit(cc_df=df1, cc_threshold = cc_thresh, figname="aa_fit.png")
    except:
        # フィッティングに失敗したことを表示
        print("fitting failed")
        # パラメータには適当な数値を入れておく
        alpha1,loc1,scale1 = 0.0, 0.0, 0.0
    # df1['cc']について mean, std, median 
    # を取得して辞書に登録
    cc_dict = {}
    cc_dict['ndata'] = df1['cc'].count()
    cc_dict['mean'] = df1['cc'].mean()
    cc_dict['std'] = df1['cc'].std()
    cc_dict['median'] = df1['cc'].median()
    cc_dict['alpha'] = alpha1
    cc_dict['loc'] = loc1
    cc_dict['scale'] = scale1

    results.append(cc_dict)
    
    print(f"AA: {len(df1)}")
    # cctypeが"BB"のものだけを抽出する
    df2 = df_filter[df_filter['cctype'] == "BB"]
    # 例外を補足する
    try:
        alpha2,loc2,scale2=fit(cc_df=df2, cc_threshold = cc_thresh, figname="bb_fit.png")
    except:
        # フィッティングに失敗したことを表示
        print("fitting failed")
        # パラメータには適当な数値を入れておく
        alpha2,loc2,scale2 = 0.0, 0.0, 0.0
        
    nAA = len(df1)
    # df1['cc']について mean, std, median 
    # を取得して辞書に登録
    cc_dict = {}
    cc_dict['ndata'] = df2['cc'].count()
    cc_dict['mean'] = df2['cc'].mean()
    cc_dict['std'] = df2['cc'].std()
    cc_dict['median'] = df2['cc'].median()
    cc_dict['alpha'] = alpha2
    cc_dict['loc'] = loc2
    cc_dict['scale'] = scale2
    
    results.append(cc_dict)

    # cctypeが"AB"のものだけを抽出する
    df3 = df_filter[df_filter['cctype'] == "AB"]
    nAA = len(df3)
    # 例外を補足する
    try:
        alpha3,loc3,scale3=fit(cc_df=df3, cc_threshold = cc_thresh, figname="ab_fit.png")
    except:
        # フィッティングに失敗したことを表示
        print("fitting failed")
        # パラメータには適当な数値を入れておく
        alpha3,loc3,scale3 = 0.0, 0.0, 0.0

    # df3['cc']について mean, std, median 
    # を取得して辞書に登録
    cc_dict = {}
    cc_dict['ndata'] = df3['cc'].count()
    cc_dict['mean'] = df3['cc'].mean()
    cc_dict['std'] = df3['cc'].std()
    cc_dict['median'] = df3['cc'].median()
    cc_dict['alpha'] = alpha3
    cc_dict['loc'] = loc3
    cc_dict['scale'] = scale3
    
    results.append(cc_dict)

    # 結果をファイルに出力 "fitting_results.txt"
    with open("fitting_results.txt", "a") as f:
        # cc_thresholdも書いておく
        f.write(f"cc_threshold: {cc_thresh}\n")
        # ヘッダー
        f.write(f"{'ndata':10s} {'mean':10s} {'std':10s} {'median':10s} {'alpha':10s} {'loc':10s} {'scale':10s}\n")
        for result in results:
            # 小数点以下は４桁まで
            f.write(f"{result['ndata']:10d} {result['mean']:.4f} {result['std']:.4f} {result['median']:.4f} {result['alpha']:.4f} {result['loc']:.4f} {result['scale']:.4f}\n")
        
    # 最後にモデルパラメータの曲線をすべて重ねたプロットを作成
    x = np.arange(0.0, 1.0, 0.001)
    y1 = skewnorm.pdf(x, alpha1, loc1, scale1)
    y2 = skewnorm.pdf(x, alpha2, loc2, scale2)
    y3 = skewnorm.pdf(x, alpha3, loc3, scale3)
    
    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y1, label="AA")
    ax.plot(x, y2, label="BB")
    ax.plot(x, y3, label="AB")
    ax.legend()
    ax.set_xlim(0.9,1.0)
    ax.set_xlabel("CC") 
    ax.set_ylabel("Probability")
    ax.set_title("CC distribution") 
    fig.savefig("cc_dist.png")