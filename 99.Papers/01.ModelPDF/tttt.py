#必要なものをインポート
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import AnaCC
# lognorm
from scipy.stats import lognorm
# skewnorm
from scipy.stats import skewnorm

#def custom_pdf(x, sigma, loc, scale, a, b):
def custom_pdf1(x, sigma, loc, scale, base):
    #baseline = a * x + b
    gauss = lognorm.pdf(1-x, sigma, loc, scale)
    #gauss = np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return gauss

def custom_pdf(x, alpha, loc, scale, a, b):
    rtn_value = skewnorm.pdf(x, alpha, loc, scale) + a*x+b
    return rtn_value

def gauss2(x, alpha, loc, scale, a2, b2, c2):
    sk_value = skewnorm.pdf(x, alpha, loc, scale) + a2 * np.exp(-((x - b2) / c2) ** 2)
    return sk_value

# CCのdataframeを受け取ってヒストグラムを返す関数
# おもにfittingのために利用する
# nbin_ratio: を設定すると、ヒストグラムのbin数を調整できる
def getHist(cc_df, cc_threshold = 0.8, nbin_ratio=8):
    # CCの数値が0.8以上のものだけに限定する
    filter_condition = cc_df['cc'] >= cc_threshold
    cc_df = cc_df[filter_condition]

    # CC data array
    ccdata = cc_df['cc']
    # binの数を計算する→改善の余地がある
    n_bins = int(len(ccdata) / nbin_ratio)

    # 初期値ヒストグラムを決定→フィッティングのクオリティはnbinsに敏感である
    hist, bin_edges = np.histogram(ccdata, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers, bin_edges

# フィッティング 
# ccの数値の配列を受け取る
def fitting(cc_df):
    # cc のヒストグラムを作成し、それにたいしてcustom_pdfをフィッティングする
    # binsの値は適当
    # ここにコードを書いてください
    # ヒストグラムを取得する
    # ヒストグラムを取得する
    hist, bin_centers, bin_edges = getHist(cc_df, 0.8, 20)
    
    # histgramを表示(帯)
    # axsのインデックスは(行、列)
    # フィッティング
    #p0 = [0.5109,-0.014,0.0149,0,0]
    #p0 = [0.5109,-0.014,0.0149,0]
    #p0 = [-15.0, 0.99, 0.02, 0.0, 0.0]
    p0 = [-15.0, 0.99, 0.02, 1.0, 0.9,0.0005]
    # boundsを指定
    #bounds = ([-30, 0.0, 0, 0.00,0.0], [0, 1.0, 0.1, 0.5, 0.01])
    #popt, pcov = curve_fit(custom_pdf, bin_centers, hist, bounds=([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, 0.05]))
    popt, pcov = curve_fit(gauss2, bin_centers, hist, p0=p0)
    # フィッティング結果の確認
    print("alpha: ", popt[0])
    print("loc: ", popt[1])
    print("scale: ", popt[2])
    print("a: ", popt[3])
    print("b: ", popt[4])

    return popt, pcov

# mainが定義されていなかったら
if __name__ == "__main__":
    import sys
    ana = AnaCC.AnaCC()
    #print(ana.read_cctable("0072"))
    print("From cluster number")
    cluster1=sys.argv[1]
    cluster2=sys.argv[2]
    # しきい値を引数から得る
    # しきい値が指定されていなければ0.8を使う
    if len(sys.argv) > 3:
        threshold = float(sys.argv[3])
    else:
        threshold = 0.9165

    # check if this function works
    error_flag = ana.crossCheck(cluster1, cluster2, cc_thresh=0.9165, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst")
    print(error_flag)

    # if error_flag is True, then there is an error
    if error_flag:
        print("Error: cross check failed")
        sys.exit(1)
    else:
        print("#################")
        df_all = ana.get_cc_various_values_from_cctable(cluster1, cluster2)
        print(df_all.head())
        # df_all の中で ctypeが"AB"であるものの抽出
        df_AB = df_all[df_all["cctype"] == "AB"]
        cc_ab = df_AB["cc"].values
        # cc_abのヒストグラムに対してスプライン補完を行う
        # cc_abのヒストグラムを作成
        hist, bin_edges = np.histogram(cc_ab, bins=30)
        # スプライン補完
        x = np.linspace(0, 1, 10000)
        # UniariateSplineを使う
        # import
        from scipy.interpolate import UnivariateSpline
        # bin_centersを計算
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        spl = UnivariateSpline(bin_centers, np.concatenate([[0], hist, [0]]) , s=0, k=3)
        # スプライン補完したものをプロット
        plt.plot(x, spl(x))
        
        # cc_abのヒストグラムを作成
        plt.hist(cc_ab, bins=100, density=False)
        plt.show()
        # cc_ab のヒストグラムの形状に合わせて、custom_pdfをフィッティング
        # フィッティング結果を表示
        popt, pcov = fitting(df_AB)
        # フィッティング結果をプロット
        x = np.linspace(0, 1, 10000)
        y_fit = gauss2(x, *popt)
        plt.plot(x, y_fit)
        # ヒストグラムを表示
        plt.hist(df_AB["cc"], bins=100, density=False)
        # y軸の範囲を指定
        plt.ylim(0, 50)
        plt.xlim(0.8,1)
        plt.show()