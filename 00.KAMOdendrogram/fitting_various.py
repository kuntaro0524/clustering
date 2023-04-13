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
        self.func_name = ["skewed gaussian", "beta pdf", "log_norm", "betaprime_pdf","sk noise"]

        self.isDebug = False

    def get_id_list_from_clusters(self, cluster_number, file_path="CLUSTERS.txt"):
        id_list = []

        print("#############")
        print(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                cols = line.split()
                #print(cols[0])
                if cols[0] == cluster_number:
                    id_list = [int(ID) - 1 for ID in cols[3:]]
                    break

        return id_list

    def get_cc_various_values_from_cctable(self, cluster_number1, cluster_number2, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst"):
        id_list1 = self.get_id_list_from_clusters(cluster_number1, clusters_file)
        id_list2 = self.get_id_list_from_clusters(cluster_number2, clusters_file)

        # isDebugがTrueの場合は、id_list1とid_list2を表示する
        if self.isDebug:
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
        self.model_funcs = [skewed_gaussian, beta_pdf, log_norm, betaprime_pdf, skewed_noise]
        #self.model_funcs = [log_norm]
        # BETA FUNCTION initial parameters for CC
        # skewed gaussian, beta pdf, log_norm, betaprime_pdf
        # def log_norm_noise(x, sigma, loc, scale, a,b):
        self.initial_params = [(-15,0.99,0.02),(1.5, 2.5, 0,1),(0.5109,-0.014,0.0149),(1.5, 2.5), (-15.0, 0.99, 0.02, 0.5, 0.0), (0.5109, -0.014, 0.0149, 0.5, 0.0)]
        #self.initial_params = [(0.5109,-0.014,0.0149)]

        return self.model_funcs

    # CCのdataframeを受け取ってヒストグラムを返す関数
    def getHist(self, cc_df, cc_threshold = 0.8, nbins=20):
        # CCの数値が0.8以上のものだけに限定する
        filter_condition = cc_df['cc'] >= cc_threshold
        cc_df = cc_df[filter_condition]

        print(f"filtered cc_df length: {len(cc_df)}")

        # CC data array
        ccdata = cc_df['cc']
        # binの数を計算する→改善の余地がある
        #n_bins = int(len(ccdata) / nbin_ratio)
        #print(n_bins)

        # 初期値ヒストグラムを決定→フィッティングのクオリティはnbinsに敏感である
        hist, bin_edges = np.histogram(ccdata, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return hist, bin_centers, bin_edges

    # CCのDataFrameを受け取って、CCの分布のヒストグラムを種々のモデル式に対してフィッティングする
    def fit(self, cc_df, cc_threshold = 0.8, nbins=20):
        # CCの数値が0.8以上のものだけに限定する
        filter_condition = cc_df['cc'] >= cc_threshold
        cc_df = cc_df[filter_condition]
        print("Filtered: ", len(cc_df), " data points")
        # CCの標準偏差,medianを計算する
        sigma_data = np.std(cc_df['cc'])
        median_data = np.median(cc_df['cc'])
        # CCの標準偏差を表示する
        print("sigma_data: ", sigma_data)
        print("median_data: ", median_data)

        # ヒストグラムを取得する
        hist, bin_centers, bin_edges = self.getHist(cc_df, cc_threshold, nbins)

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
    def run(self, clst1_name, clst2_name, cc_threshold, nbin=20):
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
        fig, axs = plt.subplots(len(model_funcs), 4, figsize=(20, 20))
        # すべてX軸は 0.8~1.0の表示する
        # for文ですべてのaxsに対して設定する
        for ax in axs.flatten():
            ax.set_xlim(0.8, 1.0)
        
        for idx,cc_df in enumerate([cc_df1, cc_df2, cc_both]):
            # ヒストグラムを取得する
            hist, bin_centers, bin_edges = self.getHist(cc_df, cc_threshold, nbin)
            plt.hist(cc_df['cc'],alpha=0.5,bins=nbin)
            plt.show()
            # フィッティングを実施
            aic_list, model_funcs, popt_list = self.fit(cc_df, cc_threshold, nbin)
            # histgramを表示(帯)
            # axsのインデックスは(行、列)

            # popt_listを表示する
            print(popt_list)
            
            # フィッティングしたモデル関数を表示
            x = np.linspace(0, 1, 1000)
            for i, func in enumerate(model_funcs):
                axs[i,idx].fill_between(bin_centers, hist, alpha=0.5)
                axs[i,idx].set_title("histgram")
                axs[i,idx].set_xlabel("CC")
                axs[i,idx].plot(x, func(x, *popt_list[i]), label=f"{func.__name__}")
                axs[i,idx].set_title(self.func_name[i])
                axs[i,idx].set_xlabel("CC")
                axs[i,idx].set_ylabel("count")
                axs[i,idx].legend()
                # fittingしたモデルパラメータを表示
                axs[i,idx].text(0.8, 0.8, f"AIC={aic_list[i]:.2f}", transform=axs[i,idx].transAxes)
                axs[i,idx].text(0.8, 0.7, f"popt={popt_list[i]}\n", transform=axs[i,idx].transAxes)

                # ３つ重ねたCC分布
                axs[i,3].fill_between(bin_centers, hist, alpha=0.5)
                

        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        plt.savefig("histgram.png")
        plt.show()

    # Arguments
    # clst1_name: cluster name 1
    # clst2_name: cluster name 2
    # cc_threshold: CC threshold
    def prepareDataframes(self, clst1_name, clst2_name, cc_threshold):
        # 1. クラスタ番号ごとにCCの分布を取得（クラス関数：get_cc_values_from_cctable）
        # 2. クラスタ番号を cluster1, cluster2 とした場合、CCの取得は cluster1単体、cluster2単体、cluster1とcluster2の合成データについて取得する
        alldf = self.get_cc_various_values_from_cctable(clst1_name, clst2_name)

        # cctypeが"AA"のもの
        df1 = alldf[alldf["cctype"] == "AA"]
        # cctypeが"BB"のもの
        df2 = alldf[alldf["cctype"] == "BB"]
        # cctypeが"AB"のもの
        df12 = alldf[alldf["cctype"] == "AB"]

        # 'cc' が cc_threshold 以上のもののみを抽出
        df1 = df1[df1["cc"] > cc_threshold]
        df2 = df2[df2["cc"] > cc_threshold]
        df12 = df12[df12["cc"] > cc_threshold]

        # それぞれのクラスタについて、データの数を表示（１行）
        # それぞれのクラスタについて、CCの平均値を表示（１行）
        print(f"cluster1: {len(df1)}")
        print(f"cluster2: {len(df2)}")
        print(f"cluster1 and cluster2: {len(df12)}")
        
        return df1,df2,df12
    
    def plotHistOnly(self, clst1_name, clst2_name, cc_threshold, nbins=20):
        df1,df2,df12 = self.prepareDataframes(clst1_name, clst2_name, cc_threshold)

        # 上記３種類のDataframeでヒストグラムを作成
        # ヒストグラムはそれぞれのdataframeごとに並べて表示する（水平方向に３つ）
        # ここではフィッティングは行わず'histgram'のみ表示する
        # forの要素ごとに１枚のfigureを作成し横に３枚並べる
        # subplotの配置を設定
        # プロット領域は４つ準備する。二段に分けて、上段はCC分布をdataframeごとにプロット（３領域）
        # 下段はCC分布を合成したものをプロット（１領域）
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))
        # すべてX軸は 0.8~1.0の表示する
        # for文ですべてのaxsに対して設定する
        #for ax in axs.flatten():
            #ax.set_xlim(0.8, 1.0)
            
        for idx, cc_df in enumerate([df1, df2, df12]):
            # ヒストグラムを取得する
            hist, bin_centers, bin_edges = self.getHist(cc_df, cc_threshold, nbins)
            #axs[1,1].bar(bin_edges[:-1], hist, width=(bin_edges[1]-bin_edges[0]), align='edge', alpha=0.5)

            # cc_dfのhistgramを表示(帯) histを利用する
            # axsのインデックスは(行、列)
            axs[0,idx].hist(cc_df['cc'], bins=nbins, alpha=0.5)
            axs[0,idx].set_title("histgram")
            axs[0,idx].set_xlabel("CC")
            axs[0,idx].set_ylabel("count")
            #axs[0,idx].set_title(f"{clst1_nameame}")
            axs[0,idx].set_xlabel("CC")
            axs[0,idx].set_ylabel("count")
            # ３つ重ねたCC分布
            axs[1,0].hist(cc_df['cc'], bins=nbins, alpha=0.5)
            axs[1,0].set_title("histgram")
            axs[1,0].set_xlabel("CC")
            axs[1,0].set_ylabel("count")
            axs[1,0].set_title("CC_both")
            axs[1,0].set_xlabel("CC")
            axs[1,0].set_ylabel("count")


        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        plt.savefig("histgram.png")
        
        plt.show()

    def runLogscale(self, clst1_name, clst2_name, cc_threshold, nbins=20):
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
        # すべてX軸は 0.8~1.0の表示する
        # for文ですべてのaxsに対して設定する
        for idx,cc_df in enumerate([cc_df1, cc_df2, cc_both]):
            fig, axs = plt.subplots(1,3, figsize=(20, 20))
            # ヒストグラムを取得する
            hist, bin_centers, bin_edges = self.getHist(cc_df, cc_threshold, nbins)
            # フィッティングを実施
            aic_list, model_funcs, popt_list = self.fit(cc_df, cc_threshold, nbins)

            # フィッティングしたモデル関数を表示
            x = np.linspace(0, 1, 1000)
            for i, func in enumerate(model_funcs):
                axs[i,idx].fill_between(bin_centers, hist, alpha=0.5)
                axs[i,idx].set_title("histgram")
                axs[i,idx].set_xlabel("CC")
                axs[i,idx].plot(x, func(x, *popt_list[i]), label=f"{func.__name__}")
                # fittingしたモデルパラメータを表示
                axs[i,idx].text(0.8, 0.8, f"AIC={aic_list[i]:.2f}", transform=axs[i,idx].transAxes)
                axs[i,idx].text(0.8, 0.7, f"popt={popt_list[i]}", transform=axs[i,idx].transAxes)
                axs[i,idx].set_title(self.func_name[i])
                axs[i,idx].set_xlabel("CC")
                axs[i,idx].set_ylabel("count")
                axs[i,idx].legend()

                # ３つ重ねたCC分布
                axs[i,3].fill_between(bin_centers, hist, alpha=0.5)

        plt.subplots_adjust(wspace=0.6, hspace=0.6)
        plt.savefig("histgram.png")
        plt.show()

if __name__ == "__main__":

    # instance
    ccModel = FittingVarious()

    cluster_numbers = sys.argv[1:]
    cc_values_list = []

    # optparseをimport
    import optparse
    # コマンドラインから以下の引数を取得する
    # option1: クラスタ番号1
    # option2: クラスタ番号2
    # option3: CCの閾値
    # option4: ヒストグラムのbin数の割合
    # optparse で引数を取得する
    # option5: 処理のタイプ
    # "histgram" : ヒストグラムのみ表示
    # "logscale" : ヒストグラムとCCの分布を表示
    
    parser = optparse.OptionParser()
    parser.add_option("-c", "--cluster1", dest="cluster1", default="0", help="cluster1")
    parser.add_option("-d", "--cluster2", dest="cluster2", default="1", help="cluster2")
    parser.add_option("-t", "--cc_threshold", dest="cc_threshold", default="0.8", help="cc_threshold")
    parser.add_option("-n", "--nbins", dest="nbins", default="20", help="nbins")
    parser.add_option("-p", "--process_type", dest="process_type", default="histogram", help="process_type")
    
    (options, args) = parser.parse_args()

    if options.process_type == "histogram":
        ccModel.plotHistOnly(options.cluster1, options.cluster2, float(options.cc_threshold), int(options.nbins))
    elif options.process_type == "logscale":
        ccModel.runLogscale(options.cluster1, options.cluster2, float(options.cc_threshold), int(options.nbins))
    elif options.process_type == "fit_various":
        ccModel.run(options.cluster1, options.cluster2, float(options.cc_threshold), int(options.nbins))