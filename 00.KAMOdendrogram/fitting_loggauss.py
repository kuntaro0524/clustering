import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
from matplotlib import pyplot as plt
from scipy.stats import beta,betaprime


class FittingVarious():
    def __init__(self):
        print("OK!")
        #self.func_name = ["skewed gaussian", "beta pdf", "log_norm", "betaprime_pdf","sk noise", "log noise"]
        self.func_name = ["log_norm"]

    # CLUSTERS.txtを読み取り、クラスター番号を指定して、そのクラスターに含まれるIDのリストを返す
    def get_id_list_from_clusters(self,cluster_number, file_path="CLUSTERS.txt"):
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

    # cctable.datを読み取り、クラスター番号を指定して、そのクラスターに含まれるCCの値のリストを返す
    def get_cc_values_from_cctable(self, cluster_number, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat"):
        id_list = self.get_id_list_from_clusters(cluster_number, clusters_file)

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
    
    # フィッティングに利用する関数の定義
    def getModelFunctions(self):
        def skewed_gaussian(x, alpha, loc, scale):
            rtn_value = skewnorm.pdf(x, alpha, loc, scale)
            return rtn_value

        def beta_pdf(x, a, b, loc, scale):
            return beta.pdf(x, a, b, loc, scale)

        from scipy.stats import lognorm
        def log_norm(x, sigma, loc, scale):
            return lognorm.pdf(1-x, sigma, loc, scale)

        # 第二種ベータ分布の確率密度関数
        def betaprime_pdf(x, alpha, beta):
            return betaprime.pdf(1-x, alpha, beta)
        
        # 一次関数をノイズモデルとしたskewed gaussian model
        def skewed_noise(x, alpha, loc, scale, a,b ):
            rtn_value = skewnorm.pdf(x, alpha, loc, scale) + a*x+b
            return rtn_value

        # 一次関数をノイズモデルとしたlognormal model
        def log_norm_noise(x, sigma, loc, scale, a,b):
            rtn_value = lognorm.pdf(1-x, sigma, loc, scale) + a*x+b
            return rtn_value

        #self.model_funcs = [skewed_gaussian, beta_pdf, log_norm, betaprime_pdf]
        #self.model_funcs = [skewed_gaussian, beta_pdf, log_norm, betaprime_pdf, skewed_noise, log_norm_noise]
        #self.model_funcs = [skewed_gaussian, beta_pdf, log_norm, betaprime_pdf, skewed_noise]
        self.model_funcs = [log_norm]
        #self.model_funcs = [log_norm]
        # BETA FUNCTION initial parameters for CC
        # skewed gaussian, beta pdf, log_norm, betaprime_pdf
        # def log_norm_noise(x, sigma, loc, scale, a,b):
        #self.initial_params = [(-15,0.99,0.02),(1.5, 2.5, 0,1),(0.5109,-0.014,0.0149),(1.5, 2.5), (-15.0, 0.99, 0.02, 0.5, 0.0), (0.5109, -0.014, 0.0149, 0.5, 0.0)]
        self.initial_params = [(0.5109,-0.014,0.0149)]

        return self.model_funcs

    # CCのdataframeを受け取ってヒストグラムを返す関数
    def getHist(self, cc_df, cc_threshold = 0.8, nbin_ratio=8):
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

        return hist, bin_centers

    # CCのDataFrameを受け取って、CCの分布のヒストグラムを種々のモデル式に対してフィッティングする
    def fit(self, cc_df, cc_threshold = 0.8, nbin_ratio=8):
        # CCの数値が0.8以上のものだけに限定する
        filter_condition = cc_df['cc'] >= cc_threshold
        cc_df = cc_df[filter_condition]
        # CCの標準偏差,medianを計算する
        sigma_data = np.std(cc_df['cc'])
        median_data = np.median(cc_df['cc'])
        # CCの標準偏差を表示する
        print("sigma_data: ", sigma_data)
        print("median_data: ", median_data)

        # ヒストグラムを取得する
        hist, bin_centers = self.getHist(cc_df, cc_threshold, nbin_ratio)

        # 初期値の設定と最小二乗法の計算
        mean = np.log(0.9)   # 対数正規分布の平均

        # model functionsを得る
        model_funcs = self.getModelFunctions()

        # 各モデル関数をフィットして、AICを計算
        aic_list = []
        popt_list = []
        for func, initial_guess,func_name in zip(model_funcs, self.initial_params,self.func_name):
            print(func_name)
            popt, pcov = curve_fit(func, bin_centers, hist, p0=initial_guess)
            popt_list.append(popt)
            residuals = hist - func(bin_centers, *popt)
            sse = np.sum(residuals**2)
            aic = 2 * len(initial_guess) + len(bin_centers) * np.log(sse / len(bin_centers))
            aic_list.append(aic)

        # aic_list, model_funcs, popt_listを返す
        return aic_list, model_funcs, popt_list
    
    # クラスター番号を２つ引数から取得して、それぞれのクラスターに含まれるCCについて以下の検討を実施する
    def run(self, clst1_name, clst2_name, cc_threshold, nbin_ratio=20):
        # 1. クラスタ番号ごとにCCの分布を取得（クラス関数：get_cc_values_from_cctable）
        # 2. クラスタ番号を cluster1, cluster2 とした場合、CCの取得は cluster1単体、cluster2単体、cluster1とcluster2の合成データについて取得する
        cc_df1 = self.get_cc_values_from_cctable(clst1_name)
        cc_df2 = self.get_cc_values_from_cctable(clst2_name)
        cc_both = pd.concat([cc_df1, cc_df2])
        
        # 上記３種類のDataframeでヒストグラムを作成
        # ヒストグラムに対して、各モデル関数をフィッティングする
        # フィッティングした結果のAICを計算する
        # ヒストグラムはそれぞれのdataframeごとに並べて表示する
        # フィッティングした結果のpoptやAICはそれぞれ該当のヒストグラム中に表示する
        # さらにフィッティングしたモデル関数もそれぞれのヒストグラム中に表示する
        # forの要素ごとに１枚のfigureを作成し横に３枚並べる
        # モデル関数ごとに縦に４行並べる
        # subplotの配置を設定
        model_funcs = self.getModelFunctions()
        fig, axs = plt.subplots(len(model_funcs), 3, figsize=(15, 5))
        fig2, axs2 = plt.subplots(len(model_funcs), 1, figsize=(20, 20))
        # すべてX軸は 0.8~1.0の表示する
        # for文ですべてのaxsに対して設定する
        for ax in axs.flatten():
            ax.set_xlim(0.8, 1.0)

        # plotのタイトルに設定するために各プロットに名称を与える
        #plot_names = [clst1_name, clst2_name, "{cls1}_{cls2}".format(cls1=clst1_name, cls2=clst2_name)]
        plot_names = ["apo-apo", "benz-benz", "apo-benz"]
        
        for idx,cc_df in enumerate([cc_df1, cc_df2, cc_both]):
            # ヒストグラムを取得する
            hist, bin_centers = self.getHist(cc_df, cc_threshold, nbin_ratio)
            # フィッティングを実施
            aic_list, model_funcs, popt_list = self.fit(cc_df, cc_threshold, nbin_ratio)
            # histgramを表示(帯)
            # axsのインデックスは(行、列)

            # グラフのタイトルを設定
            graph_title = plot_names[idx]

            # フィッティングしたモデル関数を表示
            x = np.linspace(0, 1, 1000)
            for i, func in enumerate(model_funcs):
                print("processing: ", i,idx)
                axs[idx].hist(cc_df['cc'], bins=50,alpha=0.5,density=True)
                axs[idx].plot(x, func(x, *popt_list[i]), label=f"fitted model")
                axs[idx].set_title(graph_title)
                axs[idx].set_xlabel("CC")
                axs[idx].set_ylabel("count")
                axs[idx].legend()
                # グラフ中にフィッティングの結果のパラメータを表示
                #axs[idx].text(0.9, 0.9, f"AIC: {aic_list[i]:.2f}", transform=axs[idx].transAxes)
                # poptの値を表示
                #for j, popt in enumerate(popt_list[i]):
                    #axs[idx].text(0.9, 0.9 - (j+1)*0.1, f"popt{j+1}: {popt:.5f}", transform=axs[idx].transAxes)
                #axs2.fill_between(bin_centers, hist, alpha=0.5)
                axs2.hist(cc_df['cc'], bins=20,alpha=0.5,density=True)

        fig.subplots_adjust(wspace=0.6, hspace=0.6)
        fig2.subplots_adjust(wspace=0.6, hspace=0.6)
        fig.savefig("histgram.png")
        fig2.savefig("my.png")
        plt.show()

    
    #def run(self, clst1_name, clst2_name, cc_threshold, nbin_ratio=20):

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <cluster_number1> <cluster_number2> ...")
        sys.exit(1)

    # instance
    ccModel = FittingVarious()

    cluster_numbers = sys.argv[1:]
    cc_values_list = []

    ccModel.run(cluster_numbers[0], cluster_numbers[1], 0.9165, 10)