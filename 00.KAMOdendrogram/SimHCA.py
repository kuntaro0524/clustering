# AとBの２つの構造を階層的クラスタリングで分類できるかどうかシミュレーションするために利用できるクラス
import numpy as np
import SkewedGaussianCC
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class SimHCA():
    def __init__(self, n_datasets):
        # 全データセット数
        self.n_datasetes = n_datasets
        # A構造のデータセット数
        self.n_A = int(n_datasets/2.0)
        # B構造のデータセット数
        self.n_B = self.n_datasetes - self.n_A
        self.isPrepSample=False

    # self.sample_listという配列にself.n_A, self.n_Bずつラベル"A"と"B"を追加する
    def addSampleName(self):
        self.sample_list = []
        for i in np.arange(0, self.n_A):
            self.sample_list.append("A")
        for j in np.arange(0, self.n_B):
            self.sample_list.append("B")

        self.sample_list = np.array(self.sample_list)
        self.isPrepSample=True

    # Simulationに利用するCCの配列を作成する
    # self.sample_listに登録されているデータセット（ラベル）の間で総当りでCCを計算する
    # A, Bのリストについては addSampleNameで作成したself.sample_listに含まれる総当りの組み合わせで計算する
    # この場合にはCCはAAの分布に従うため、AAの分布からCCを計算する
    # CCの分布はA-A, A-B, B-Bの３つに分類されるため、それぞれの分布を作成する
    # この関数では特にskewed gaussian分布のパラメータである、alpha, loc, scaleを３対受け取る
    # それぞれ受け取ったパラメータを用いて、A-A, A-B, B-BのCCの分布を作成し、ccAA, ccBB, ccABというnumpy配列に格納
    # CC計算はSkewedGaussianCCクラスのrvs関数を利用する
    # さらに同時に、A-A, A-B, B-BのCCの分布を作成するために必要な、A-A, A-B, B-Bの組み合わせのラベルをself.name_listに格納
    # 例えばA-AのCC計算の結果であれば、ccAAにはA-AのCCの分布が、name_listにはA-Aの組み合わせのラベルが格納される
    # skewednormal分布の戻り値が1.0をこえる場合は、1.0に置き換える
    # 最終的にクラスタリングに必要な距離行列の計算も実施する。
    # distance = sqrt(1-cc^2)
    # によって計算し、self.distance_matrixという名前のnumpy配列に格納する dimensionは1次元
    def makeCC(self, aa_sets, bb_sets, ab_sets):
        from scipy.stats import skewnorm
        import numpy as np
        import SkewedGaussianCC

        # aa_sets, bb_sets, ab_sets にはそれぞれ(alpha, loc, scale)のタプルが格納されている
        # この関数内で利用できるように展開する
        alphaAA, locAA, scaleAA = aa_sets
        alphaBB, locBB, scaleBB = bb_sets
        alphaAB, locAB, scaleAB = ab_sets

        self.ccAA = []
        self.ccAB = []
        self.ccBB = []
        self.distance_matrix = []

        for idx1,s1 in enumerate(self.sample_list):
            for s2 in self.sample_list[idx1+1:]:
                if s1=="A" and s2=="A":
                    cctmp = SkewedGaussianCC.SkewedGaussianCC(alphaAA, locAA, scaleAA).rvs()
                elif s1=="B" and s2=="B":
                    cctmp = SkewedGaussianCC.SkewedGaussianCC(alphaBB, locBB, scaleBB).rvs()
                    self.ccBB.append(cctmp[0])
                else:
                    cctmp = SkewedGaussianCC.SkewedGaussianCC(alphaAB, locAB, scaleAB).rvs()
                    self.ccAB.append(cctmp[0])

                if cctmp[0]>1.0:
                    cctmp[0]=1.0

                # distance = sqrt(1-cc^2)
                distance = np.sqrt(1-cctmp[0]*cctmp[0])
                self.distance_matrix.append(distance)

    # 階層的クラスタリングを実施する
    # この関数では、self.distance_matrixを利用して階層的クラスタリングを実施する
    # scipy hierarchical clusteringを利用する
    # Ward法によるクラスタリングを実施し、self.Zに格納する
    # また得られた結果のself.Zからデンドロぐラムを作成し、self.dendrogramに格納する
    def calcZ(self, isDraw=False):
        from scipy.cluster.hierarchy import dendrogram, linkage
        from matplotlib import pyplot as plt

        self.Z = linkage(self.distance_matrix, 'ward')

        if isDraw:
            self.dendrogram = dendrogram(self.Z, labels=self.sample_list,leaf_font_size=8)
            #デンドログラムを表示する
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('sample index')
            plt.ylabel('distance')
            plt.show()

        return self.Z
    
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
    def evaluateClustering(self, threshold):
        from scipy.cluster.hierarchy import fcluster
        import numpy as np

        cluster_indices = fcluster(self.Z, threshold, criterion='distance')
        num_clusters = len(np.unique(cluster_indices))

        if num_clusters != 2:
            print(num_clusters)
            return [False, 0, 0, 0, 0, 0]

        cluster_1_indices = np.where(cluster_indices == 1)[0]
        cluster_2_indices = np.where(cluster_indices == 2)[0]

        #print(cluster_1_indices)
        cluster_1_label = self.sample_list[cluster_1_indices]
        cluster_2_label = self.sample_list[cluster_2_indices]

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

    # 結果の評価のための作図を行うクラス関数
    # 入力は、AA, AB, BBのCC計算に利用した歪ガウス関数のパラメータのリスト
    # クラスタリングの結果についてはself.Zを利用する
    # alpha_aa, loc_aa, scale_aa, alpha_ab, loc_ab, scale_ab, alpha_bb, loc_bb, scale_bb
    # クラスタリングで得られた結果のZと、クラスタリングの対象となったデータセットのラベルのリスト、しきい値である
    # プロットには２つの領域を用意する
    # 左側の1/3には歪ガウス関数の形状をプロットする。AA, AB, BBの順にプロットする
    # 歪ガウス関数には SkewedGaussianクラスのget_mode関数を利用した最大値の取得のあと、x_maxとして、透明度0.2でx=x_maxを破線で描く
    # 右側の2/3にはデンドログラムをプロットする
    def plotClustering(self, fig_name, aa_sets, bb_sets, ab_sets, threshold):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        # 歪ガウス関数のパラメータは(alpha, loc, scale)の順に格納されている
        # aa_sets, bb_sets, ab_setsがそれに相当する
        alpha_aa, loc_aa, scale_aa = aa_sets
        alpha_ab, loc_ab, scale_ab = ab_sets
        alpha_bb, loc_bb, scale_bb = bb_sets

        fig = plt.figure(figsize=(24, 10))
        # ax1とax2に分けるが、大きさは1/3と2/3にする。ax1は左側、ax2は右側
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, (2, 3))

        x = np.arange(0, 1, 0.001)
        ax1.plot(x, SkewedGaussianCC.SkewedGaussianCC(alpha_aa, loc_aa, scale_aa).pdf(x), label="AA", color="orange")
        ax1.plot(x, SkewedGaussianCC.SkewedGaussianCC(alpha_ab, loc_ab, scale_ab).pdf(x), label="AB", color="green")
        ax1.plot(x, SkewedGaussianCC.SkewedGaussianCC(alpha_bb, loc_bb, scale_bb).pdf(x), label="BB", color="blue")
        ax1.legend()

        # 歪ガウス関数の最大値を取得し、破線で描く
        aa_max = SkewedGaussianCC.SkewedGaussianCC(alpha_aa, loc_aa, scale_aa).get_mode()
        bb_max = SkewedGaussianCC.SkewedGaussianCC(alpha_bb, loc_bb, scale_bb).get_mode()
        ab_max = SkewedGaussianCC.SkewedGaussianCC(alpha_ab, loc_ab, scale_ab).get_mode()

        # 歪ガウス関数ごとにalpha, loc, scale, maxの数値(例：aa_max) をannotateを利用して図中にコメント追記する
        ax1.annotate(f"aa_max={aa_max:.4f}", xy=(0.2, 0.90), xycoords='axes fraction')
        ax1.annotate(f"bb_max={bb_max:.4f}", xy=(0.2, 0.88), xycoords='axes fraction')
        ax1.annotate(f"ab_max={ab_max:.4f}", xy=(0.2, 0.86), xycoords='axes fraction')
        ax1.annotate(f"alpha_aa={alpha_aa:.4f}", xy=(0.2, 0.80), xycoords='axes fraction')
        ax1.annotate(f"alpha_ab={alpha_ab:.4f}", xy=(0.2, 0.78), xycoords='axes fraction')
        ax1.annotate(f"alpha_bb={alpha_bb:.4f}", xy=(0.2, 0.76), xycoords='axes fraction')
        ax1.annotate(f"loc_aa={loc_aa:.4f}", xy=(0.2, 0.70), xycoords='axes fraction')
        ax1.annotate(f"loc_ab={loc_ab:.4f}", xy=(0.2, 0.68), xycoords='axes fraction')
        ax1.annotate(f"loc_bb={loc_bb:.4f}", xy=(0.2, 0.66), xycoords='axes fraction')
        ax1.annotate(f"scale_aa={scale_aa:.4f}", xy=(0.2, 0.60), xycoords='axes fraction')
        ax1.annotate(f"scale_ab={scale_ab:.4f}", xy=(0.2, 0.58), xycoords='axes fraction')
        ax1.annotate(f"scale_bb={scale_bb:.4f}", xy=(0.2, 0.56), xycoords='axes fraction')

        # 横軸と縦軸のラベルを入れる
        ax1.set_xlabel("CC")
        ax1.set_ylabel("Probability")

        # 歪ガウス関数の最大値をY軸に平行な直線として描く
        # 色はAA, AB, BBの歪ガウス関数の色と合わせる
        # axvlineを利用する
        ax1.axvline(aa_max, color="orange", linestyle="--", alpha=0.2)
        ax1.axvline(ab_max, color="green", linestyle="--", alpha=0.2)
        ax1.axvline(bb_max, color="blue", linestyle="--", alpha=0.2)

        # ax1のタイトルを設定する
        ax1.set_title("Skewed Gaussian Distribution")
        # ax2のタイトルを設定する
        ax2.set_title("Dendrogram")

        # ２グループに分かれたときの最後の結合を取得し、Ward距離を取得
        last_merge = self.Z[-2]
        thresh_obs = last_merge[2]

        # ax1のXの範囲は0.9-1.0にする
        ax1.set_xlim(0.9, 1.0)
        dendrogram(self.Z, labels=self.sample_list, ax=ax2, leaf_rotation=90, leaf_font_size=10)
        # デンドログラム内部に thresh_obsの数値を文字列で表示する
        # thresh_obsはfloat型なので、str関数で文字列に変換する。フォーマットは{.3f}
        # ax2の水平方向の中央あたりに表示
        # annotateを利用
        ax2.annotate(f"Threshold for two main clusters: {thresh_obs:.3f}", xy=(0.5, 0.70), xycoords='axes fraction')
        plt.savefig(fig_name)
        # plt をクリアしておく
        plt.clf()
        #plt.show()

    # 他のクラス関数を活用してクラスタリングの計算、評価、プロットを行うクラス関数
    def doClustering(self, threshold):
        shca.addSampleName()
        aa_sets = (-15.2177, 0.9945, 0.0195)
        bb_sets = (-8.475, 0.9883, 0.0251)
        ab_sets = (-11.3898, 0.9799, 0.0277)
        self.makeCC(aa_sets, bb_sets, ab_sets)
        self.calcZ()
        self.plotClustering("test.png", aa_sets, bb_sets, ab_sets, threshold )

        return self.evaluateClustering(threshold)

    # skewed gaussian パラメータを変更し、クラスタリングの結果から分類の成功の是非を判定するクラス関数
    # この関数ではABのlocのみを変更した場合のクラスタリングの結果を評価する
    # 具体的な手順は以下
    # 1. loc_abの範囲を指定する
    # 2. loc_abの範囲をfor文で回す
    # 3. loc_abの値は小さい方から大きい方へ0.01ずつ変化させる
    # 3. loc_abの値を変更して、クラスタリングを行う
    # 4. クラスタリングの結果から、分類の成功の是非を判定する
    # 5. すでに完成しているクラス関数self.evaluate_clusteringの結果 cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count を評価する
    # 6. 分類成功の定義は?
    # 7. loc_abの値と、分類の成功の是非をリストに格納する
    # 8. 分類に成功する場合と失敗する場合の境界を求める
    # n_timesは計算の繰り返し回数を指定する
    # thresholdはクラスタリングの閾値を指定する
    # この関数は、loc_abの値を変更して、クラスタリングを行い、分類の成功の是非を判定する
    # loc_abの範囲は min_loc_ab から max_loc_ab まで step_loc_ab ずつ変化させる
    def simSmallerDiff(self, min_loc_ab, max_loc_ab, step_loc_ab, n_times, threshold):
        self.boundary_list = []

        if self.isPrepSample == False:
            self.addSampleName()
        
        for ith_cycle in range(n_times):
            # ファイルのプレフィクスはcycle01, cycle02, ...とする
            file_prefix = f"cycle{ith_cycle:02d}"
            # 荒いスキャンを実施する
            # このときのfile_prefixはcycle01_coarse
            file_prefix_coarse = file_prefix + "_coarse"
            boundary_ab, results1 = self.simChangeLocAB(min_loc_ab, max_loc_ab, step_loc_ab, threshold, file_prefix_coarse)
            # 細かいスキャンを実施する
            # このときのfile_prefixはcycle01_fine
            file_prefix = file_prefix + "_fine"
            # boundary_ab の周囲を少し細かく解析する　self.simChangeLocAB関数を利用
            # fine_stepはstep_loc_abの1/10
            # fine_rangeはfine_stepの10倍
            fine_step = step_loc_ab / 10.0
            fine_range = fine_step * 10.0
            min_loc_ab = boundary_ab - fine_range 
            max_loc_ab = boundary_ab + fine_range
            boundary_fine, results2 = self.simChangeLocAB(min_loc_ab, max_loc_ab, fine_step, threshold, file_prefix)
            
            # results1, results2の結果をpandasの別個のdataframeに格納する
            df_tmp1 = pd.DataFrame(results1, columns=["dataindex", "loc_ab", "cluster_1_A_purity", "cluster_1_B_purity", "cluster_2_A_purity", "cluster_2_B_purity", "cluster_1_count", "cluster_2_count"])
            df_tmp2 = pd.DataFrame(results2, columns=["dataindex", "loc_ab", "cluster_1_A_purity", "cluster_1_B_purity", "cluster_2_A_purity", "cluster_2_B_purity", "cluster_1_count", "cluster_2_count"])
            # df_tmp1 の 'type' カラムに'coarse'を格納する
            df_tmp1['type'] = 'coarse'
            # df_tmp2 の 'type' カラムに'fine'を格納する
            df_tmp2['type'] = 'fine'
            # df_tmp1, df_tmp2を結合する
            df_tmp = pd.concat([df_tmp1, df_tmp2])
            # df_tmp の 'cycle' という列に、ith_cycleの値を格納する
            df_tmp['cycle'] = ith_cycle
            # すでに計算した結果がある場合は、結果を結合する
            if ith_cycle == 0:
                df = df_tmp
            else:
                df = pd.concat([df, df_tmp])
            
            # boundary_fineの値をboundary_listに格納する
            self.boundary_list.append(boundary_fine)

        # dfの結果をCSVファイルとして出力する
        df.to_csv("simSmallerDiff.csv")

        # self.boundary_listに含まれるboundary_fineの値の平均と標準偏差を求める
        boundary_mean = np.mean(self.boundary_list)
        boundary_std = np.std(self.boundary_list)

        print(f"boundary_mean = {boundary_mean}")
        print(f"boundary_std = {boundary_std}")

        # Log fileに結果を出力する
        # boundary_mean, boundary_stdを出力する
        with open("final.dat", 'a') as f:
            f.write(f"boundary_mean = {boundary_mean} boundary_std = {boundary_std}\n")

    def obsoleted():
        # results1 と results2の結果を、loc_abの値の昇順にソート
        # sorted関数を利用
        # CSVファイルとして出力する
        # CSVファイルのファイル名は、"simSmallerDiff.csv"とする
        # CSVファイルのヘッダーは、"dataindex, loc_ab, cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count"とする
        # CSVファイルのデータは、results1とresults2を結合したものとする
        # x[1] : loc_abの値
        sorted_results = sorted(results1 + results2, key=lambda x: x[1])
        with open("simSmallerDiff.csv", "w") as f:
            f.write("dataindex, loc_ab, cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count\n")
            for idx, result in enumerate(sorted_results):
                # 出力フォーマット cluster_1_count, cluster_2_countは整数型
                # 出力フォーマット cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purityは小数点以下4桁
                f.write(f"{result[0]:3d}, {result[1]:.3f}, {result[2]:.4f}, {result[3]:.4f}, {result[4]:.4f}, {result[5]:.4f}, {int(result[6])}, {int(result[7])}\n")

        # 結果のプロットを作成する
        # x軸はloc_abの値
        # y1軸は、cluster_1_count, cluster_2_countの値
        # y2軸は purityの値
        # すべてのプロットは、1つのグラフに表示する
        # グラフのタイトルは、"simSmallerDiff"とする
        # グラフのファイル名は、"simSmallerDiff.png"とする
        # グラフのx軸のラベルは、"loc_ab"とする
        # グラフのy軸のラベルは、"purity, count"とする
        # グラフの凡例は、"cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count"とする
        # グラフの線の色は、"red, blue, green, yellow, black, gray"とする
        # グラフの線のスタイルは、"solid, solid, solid, solid, solid, solid"とする
        # グラフの線の太さは、"1, 1, 1, 1, 1, 1"とする
        plt.figure(figsize=(25, 25))
        plt.subplot(2, 1, 1)
        plt.title("simSmallerDiff")
        plt.xlabel("loc_ab")
        plt.ylabel("purity")
        # pointで表示する lineは不要
        plt.scatter([result[1] for result in sorted_results], [result[2] for result in sorted_results], color="red", linestyle="solid", linewidth=1, marker="o")
        plt.scatter([result[1] for result in sorted_results], [result[3] for result in sorted_results], color="blue", linestyle="solid", linewidth=1, marker="o")
        plt.scatter([result[1] for result in sorted_results], [result[4] for result in sorted_results], color="green", linestyle="solid", linewidth=1, marker="o")
        plt.scatter([result[1] for result in sorted_results], [result[5] for result in sorted_results], color="orange", linestyle="solid", linewidth=1, marker="o")
        plt.legend(["cluster_1_A_purity", "cluster_1_B_purity", "cluster_2_A_purity", "cluster_2_B_purity"])
        plt.subplot(2, 1, 2)
        plt.xlabel("loc_ab")
        plt.ylabel("count")
        plt.plot([result[1] for result in sorted_results], [result[6] for result in sorted_results], color="black", linestyle="solid", linewidth=1, marker="o")
        plt.plot([result[1] for result in sorted_results], [result[7] for result in sorted_results], color="gray", linestyle="solid", linewidth=1, marker="o")
        plt.legend(["cluster_1_count", "cluster_2_count"])
        plt.savefig("simSmallerDiff.png")   

    def simChangeLocAB(self, min_loc, max_loc, step, threshold, prefix="test"):
        # loc_abの範囲を指定する
        loc_ab_range = np.arange(min_loc, max_loc, step)

        # simulation計算の結果を格納するリスト
        loc_sim_results = []
        for didx, loc_ab in enumerate(loc_ab_range):
            print(f"loc_ab: {loc_ab}")
            # loc_abの値を変更して、クラスタリングを行う
            aa_sets = (-15.2177, 0.9945, 0.0195)
            bb_sets = (-8.475, 0.9883, 0.0251)
            ab_sets = (-11.3898, loc_ab, 0.0277)
            self.makeCC(aa_sets, bb_sets, ab_sets)
            self.calcZ()
            # Figure name for this index
            figname = f"{prefix}_{didx:03d}.png"
            self.plotClustering(figname, aa_sets, bb_sets, ab_sets, threshold)
            # クラスタリングの結果から、分類の成功の是非を判定する
            results = self.evaluateClustering(threshold)
            if results[0] != False:
                cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count = results
            else:
                print("Failed to classify")
                failure_loc_ab = loc_ab
                break
                #continue
            # loc_abの値と、分類の成功の是非をリストに格納する
            loc_sim_results.append([didx, loc_ab, cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count])

            # 分類の成功の是非を判定する
            # ここではcluster_1_count, cluster_2_count を評価する
            # cluster_1_count は正常に分類されたら self.n_A に等しくなる
            # cluster_2_count は正常に分類されたら self.n_B に等しくなる
            # 厳密には、cluster_1_count は self.n_A に等しくなるとは限らない（なぜならば cluster_1_countがBの数になる可能性があるから）
            # cluster_1_count が self.n_A に等しくなる場合は、cluster_2_count は self.n_B に等しくなる
            # いずれにせよここでは、self.n_Aの数の20%以上の数がcluster_1_count, もしくは cluster_2_count ある場合には分類に失敗したと判定する
            if (cluster_1_count > self.n_A * 1.2) or (cluster_2_count > self.n_A * 1.2):
                print("Too many datasets in one cluster.")
                print(cluster_1_count, cluster_2_count)
                # 分類に失敗する場合のloc_abの値を記録する
                failure_loc_ab = loc_ab
                break
            else:
                # 分類に成功する場合のloc_abの値を記録する
                success_loc_ab = loc_ab

            # 分類の成功の是非を判定する
            if (cluster_1_A_purity >= 0.8 and cluster_1_B_purity <= 0.2 and cluster_2_A_purity <= 0.2 and cluster_2_B_purity >= 0.8) or \
                (cluster_1_A_purity <= 0.2 and cluster_1_B_purity >= 0.8 and cluster_2_A_purity >= 0.8 and cluster_2_B_purity <= 0.2):
                # 分類に成功する場合のloc_abの値を記録する
                success_loc_ab = loc_ab
            else:
                print("Failed.")
                # 分類に失敗する場合のloc_abの値を記録する
                failure_loc_ab = loc_ab
                break
        
        # 分類に成功する場合と失敗する場合の境界を求める
        boundary_loc_ab = (success_loc_ab + failure_loc_ab) / 2
        print(f"boundary_loc_ab: {boundary_loc_ab:.3f}")
        return boundary_loc_ab, loc_sim_results

# テスト用のmain関数を定義する
if __name__ == "__main__":
    shca=SimHCA(100)
    #def simSmallerDiff(self, min_loc_ab, max_loc_ab, step_loc_ab, n_times, threshold):
    shca.simSmallerDiff(0.97, 1.00, 0.003, 10, 0.6)
    