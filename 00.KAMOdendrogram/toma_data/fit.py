import sys
import pandas as pd
import numpy as np
# scipy curve_fit をインポート
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
from matplotlib import pyplot as plt
from scipy.stats import beta

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


def get_cc_values_from_cctable(cluster_number, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat"):
    id_list = get_id_list_from_clusters(cluster_number, clusters_file)

    cc_values = []

    with open(cctable_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            cols = line.split()
            i, j = int(cols[0]), int(cols[1])

            if i in id_list and j in id_list:
                cc_values.append(float(cols[2]))

    # cc_values が格納されたDataFrameを返す
    ret = pd.DataFrame(cc_values, columns=["cc"])
    
    return ret

# CCのDataFrameを受け取って、CCの分布をヒストグラムをskeewed gaussianでフィッティングする
def fit(cc_df, cc_threshold = 0.8):
    # CCの数値が0.8以上のものだけに限定する
    filter_condition = cc_df['cc'] >= cc_threshold
    cc_df = cc_df[filter_condition]
    # CCの標準偏差を計算する
    sigma_data = np.std(cc_df['cc'])
    # CCの標準偏差を表示する
    print("sigma_data: ", sigma_data)
    # CC data array
    ccdata = cc_df['cc']
    n_bins = int(len(ccdata) / 8)
    print(n_bins)

    # 初期値？
    hist, bin_edges = np.histogram(ccdata, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # skewed gaussian 関数の定義
    def skewed_gaussian(x, alpha, loc, scale):
        rtn_value = skewnorm.pdf(x, alpha, loc, scale)
        return rtn_value

    def beta_pdf(x, a, b, loc, scale):
        return beta.pdf(x, a, b, loc, scale)

    from scipy.stats import lognorm
    def log_norm(x, sigma, loc, scale):
        return lognorm.pdf(1-x, sigma, loc, scale)

    from scipy.stats import betaprime
    def betaprime_pdf(x,alpha,beta):
        return betaprime.pdf(x,alpha,beta)

    # 初期値の設定と最小二乗法の計算
    mean = np.log(0.9)   # 対数正規分布の平均

    initial_guess = [1.5, np.log(0.9), np.exp(mean)]
    popt, pcov = curve_fit(log_norm, bin_centers, hist, p0=initial_guess)
    alpha_fit, loc_fit, scale_fit = popt

    # Histogram drawing
    plt.hist(ccdata, bins=n_bins, color='green', alpha=0.5)
    plt.xlabel('CC')
    plt.ylabel('Frequency')

    # Plotting the resultant skewed gaussian function
    xrange=np.arange(0.1,1,0.001)
    plt.plot(xrange, log_norm(xrange, *popt), 'r-', label='Fitted Skew Gaussian')
    plt.xlim(0.8,1)

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

    for idx,cluster_number in enumerate(cluster_numbers):
        ccdf = get_cc_values_from_cctable(cluster_number)
        if idx==0:
            rtn_df = ccdf
        else:
            rtn_df = pd.concat([rtn_df, ccdf], axis=0)

    #print(f"CC values for cluster {cluster_number}: {cc_values}")
    fit(cc_df=rtn_df, cc_threshold = 0.6)

    cc_data = pd.DataFrame({'Cluster_Number': cluster_numbers, 'CC_Values': cc_values_list})
    print(cc_data)
