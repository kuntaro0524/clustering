# AとBの２つの構造を階層的クラスタリングで分類できるかどうかシミュレーションするために利用できるクラス
import numpy as np
import SkewedGaussianCC
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

class SimHCA():
    def __init__(self, n_datasets, aa_sets, bb_sets, ab_sets):
        # 全データセット数
        self.n_datasetes = n_datasets
        # A構造のデータセット数
        self.n_A = int(n_datasets/2.0)
        # B構造のデータセット数
        self.n_B = self.n_datasetes - self.n_A
        self.isPrepSample=False
        # simulation で利用するskewed gaussian関数のパラメータをセットする
        # この関数では、A-A, A-B, B-Bの３つの分布のパラメータをセットする
        # パラメータは各関数でalpha, loc, scaleの３つである。
        # それぞれのパラメータはdict型の変数とし、"AA", "AB", "BB"のキーでアクセスできるようにする
        # 受け取る aa_setsには (alpha, loc, scale)が入っているのでそれぞれの値をキーでアクセスできるようにする
        self.sample_dict = [{"name":"AA", "alpha":aa_sets[0],"loc":aa_sets[1],"scale":aa_sets[2]},
                            {"name":"AB", "alpha":ab_sets[0],"loc":ab_sets[1],"scale":ab_sets[2]},
                            {"name":"BB", "alpha":bb_sets[0],"loc":bb_sets[1],"scale":bb_sets[2]}]

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
    def makeCC(self):
        from scipy.stats import skewnorm
        import numpy as np
        import SkewedGaussianCC

        # aa_sets, bb_sets, ab_sets にはそれぞれ(alpha, loc, scale)のタプルが格納されている
        # この関数内で利用できるように展開する
        alphaAA, locAA, scaleAA = self.getParamDataAxis("AA")
        alphaBB, locBB, scaleBB = self.getParamDataAxis("BB")
        alphaAB, locAB, scaleAB = self.getParamDataAxis("AB")

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
    def plotClustering(self, fig_name, threshold):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram

        # 歪ガウス関数のパラメータは(alpha, loc, scale)の順に格納されている
        # aa_sets, bb_sets, ab_setsがそれに相当する
        # self.sample_dictはdictの配列で、keyは "name", "alpha", "loc", "scale" である
        # それぞれのkeyに対応する値を取り出す
        # 例えば self.sample_dictのnameが"AA"の場合、そのalpha, loc, scaleはalpha_aa, loc_aa, scale_aaに格納する
        for sample in self.sample_dict:
            if sample["name"] == "AA":
                alpha_aa = sample["alpha"]
                loc_aa = sample["loc"]
                scale_aa = sample["scale"]
            elif sample["name"] == "AB":
                alpha_ab = sample["alpha"]
                loc_ab = sample["loc"]
                scale_ab = sample["scale"]
            elif sample["name"] == "BB":
                alpha_bb = sample["alpha"]
                loc_bb = sample["loc"]
                scale_bb = sample["scale"]

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
    def simSmallerDiff(self, data="AA",param="loc",min_value=0.0, max_value=1.0, step=0.01, n_times=10, threshold=0.6):
        self.boundary_list = []

        if self.isPrepSample == False:
            self.addSampleName()
        
        for ith_cycle in range(n_times):
            # ファイルのプレフィクスはcycle01, cycle02, ...とする
            file_prefix = f"cycle{ith_cycle:02d}"
            # 荒いスキャンを実施する
            # このときのfile_prefixはcycle01_coarse
            # この結果、boundary_ab, results1 が得られる
            file_prefix_coarse = file_prefix + "_coarse"
            boundary_coarse, results1 = self.simChangeAndGo(data, param, min_value, max_value, step, threshold, file_prefix_coarse)
            # boundary_abにはNoneが入っている可能性があるので、Noneが入っていた場合にはこのサイクルの残りの処理はスキップする
            if boundary_coarse is None:
                continue
            # 細かいスキャンを実施する
            # このときのfile_prefixはcycle01_fine
            file_prefix_fine = file_prefix + "_fine"
            # boundary_ab の周囲を少し細かく解析する　self.simChangeLocAB関数を利用
            # fine_stepはstep_loc_abの1/10
            # fine_rangeはfine_stepの10倍
            fine_step = step / 10.0
            fine_range = fine_step * 10.0
            min_fine = boundary_coarse - fine_range 
            max_fine = boundary_coarse + fine_range
            boundary_fine, results2 = self.simChangeAndGo(data, param, min_fine, max_fine, fine_step, threshold, file_prefix_fine)
            
            # results1, results2の結果をpandasの別個のdataframeに格納する
            df_tmp1 = pd.DataFrame(results1, columns=["dataindex", "param", "cluster_1_A_purity", "cluster_1_B_purity", "cluster_2_A_purity", "cluster_2_B_purity", "cluster_1_count", "cluster_2_count","isomorphic_threshold","fig_name"])
            df_tmp2 = pd.DataFrame(results2, columns=["dataindex", "param", "cluster_1_A_purity", "cluster_1_B_purity", "cluster_2_A_purity", "cluster_2_B_purity", "cluster_1_count", "cluster_2_count","isomorphic_threshold","fig_name"])
            # df_tmp1 の 'type' カラムに'coarse'を格納する
            df_tmp1['type'] = 'coarse'
            # df_tmp1の'boundary'カラムにboundary_coarseの値を格納する
            df_tmp1['boundary'] = boundary_coarse
            # df_tmp2 の 'type' カラムに'fine'を格納する
            df_tmp2['type'] = 'fine'
            # df_tmp2 の 'boundary' カラムにboundary_fineの値を格納する
            df_tmp2['boundary'] = boundary_fine
            # df_tmp1, df_tmp2を結合する
            df_tmp = pd.concat([df_tmp1, df_tmp2])
            # dfに data, param の名前を格納する
            df_tmp['data'] = data
            df_tmp['param'] = param
            # df_tmp の 'cycle' という列に、ith_cycleの値を格納する
            df_tmp['cycle'] = ith_cycle
            # df_tmpのカラムを並び替える。具体的には、'data', 'param', 'cycle', 'type', 'fig_name', 'boundary', 'dataindex', 'param', 'cluster_1_A_purity', 'cluster_1_B_purity', 'cluster_2_A_purity', 'cluster_2_B_purity', 'cluster_1_count', 'cluster_2_count','isomorphic_threshold' の順に並び替える
            df_tmp = df_tmp[['data', 'param', 'cycle', 'type', 'fig_name', 'boundary', 'dataindex', 'cluster_1_A_purity', 'cluster_1_B_purity', 'cluster_2_A_purity', 'cluster_2_B_purity', 'cluster_1_count', 'cluster_2_count','isomorphic_threshold']]
            # すでに計算した結果がある場合は、結果を結合する
            if ith_cycle == 0:
                df = df_tmp
            else:
                df = pd.concat([df, df_tmp])
            
            # boundary_fineの値をboundary_listに格納する
            self.boundary_list.append(boundary_fine)

        # もしも self.boundary_list が空の場合はコメントを出力して終了する
        if len(self.boundary_list) == 0:
            print("self.boundary_list is empty")
            return
        
        # dfの結果をCSVファイルとして出力する
        df.to_csv("simSmallerDiff.csv")

        # self.boundary_listに含まれるboundary_fineの値の平均と標準偏差を求める
        boundary_mean = np.mean(self.boundary_list)
        boundary_std = np.std(self.boundary_list)

        print(f"boundary_mean = {boundary_mean}")
        print(f"boundary_std = {boundary_std}")

        # Log fileに結果を出力する
        # boundary_mean, boundary_stdを出力する
        # boundary_meanに計算したデータ数も出力する
        # modelのパラメータも出力する (self.aa_sets, self.bb_sets, self.ab_sets)
        with open("final.dat", 'w') as f:
            f.write(f"boundary_mean = {boundary_mean} ({len(self.boundary_list)})\n")
            f.write(f"boundary_std = {boundary_std}\n")
            # AA のパラメータを出力する self.getParamDataAxis("AA") は、self.sample_dictの中から、AAに対応する値を取り出す
            f.write(f"AA = {self.getParamDataAxis('AA')}\n")
            # BB のパラメータを出力する self.getParamDataAxis("BB") は、self.sample_dictの中から、BBに対応する値を取り出す
            f.write(f"BB = {self.getParamDataAxis('BB')}\n")
            # AB のパラメータを出力する self.getParamDataAxis("AB") は、self.sample_dictの中から、ABに対応する値を取り出す
            f.write(f"AB = {self.getParamDataAxis('AB')}\n")

    def simChangeAndGo(self, data, param, min_loc, max_loc, step, threshold, prefix="test"):
        #　data名、param名の数値を少しずつ変更しながら、simulation計算を行う
        # self.sample_dictの中から、data名、param名に対応する値を取り出す
        # その値を、min_locからmax_locまで、stepごとに変更しながら、simulation計算を行う
        scan_range = np.arange(min_loc, max_loc, step)

        # simulationCoreを呼び出して、simulation計算を行う
        success_loc_ab, failure_loc_ab, loc_sim_results = self.simulationCore(data, param, scan_range, threshold, prefix)

        # succcess_loc_ab, failure_loc_abの値がいずれか、もしくはいずれもNoneの場合には、boundary_loc_abを求めることができない
        if success_loc_ab is None or failure_loc_ab is None:
            print("boundary cannot be calculated")
            # success_loc_ab, failure_loc_abは一応表示しておく
            print(f"success_loc_ab: {success_loc_ab}")
            print(f"failure_loc_ab: {failure_loc_ab}")
            return None, None
        else:
            # 分類に成功する場合と失敗する場合の境界を求める
            boundary_loc_ab = (success_loc_ab + failure_loc_ab) / 2
            print(f"boundary_loc_ab: {boundary_loc_ab:.3f}")
            return boundary_loc_ab, loc_sim_results
    
    # self.sample_dictの中で指定したdata名のパラメータを変更するクラス関数
    def setParamDataAxis(self, data, param, value):
        # 歪ガウス関数のパラメータは(alpha, loc, scale)の順に格納されている
        # aa_sets, bb_sets, ab_setsがそれに相当する
        # self.sample_dictはdictの配列で、keyは "name", "alpha", "loc", "scale" である
        # それぞれのkeyに対応する値を取り出す
        # 例えば self.sample_dictのnameが"AA"の場合、そのalpha, loc, scaleはalpha_aa, loc_aa, scale_aaに格納する
        print(f"{data, param, value}")
        for sample in self.sample_dict:
            if sample["name"] == data:
                sample[param] = value

    # self.sample_dictの中から指定されたdata名のパラメータを取り出すクラス関数
    # 例えば self.sample_dictのnameが"AA"の場合、そのalpha, loc, scaleをタプルとして返す
    def getParamDataAxis(self, data):
        for sample in self.sample_dict:
            if sample["name"] == data:
                return sample["alpha"], sample["loc"], sample["scale"]

    # simulation coreパートを抜き出して定義する
    def simulationCore(self, data, param, scan_range, threshold, prefix="test"):
        # simulation計算の結果を格納するリスト
        loc_sim_results = []
        for didx, current_value in enumerate(scan_range):
            # 指定したデータの指定したパラメータの値を変更して、クラスタリングを行う
            self.setParamDataAxis(data, param, current_value)
            self.makeCC()
            self.calcZ()
            # Figure name for this index
            figname = f"{prefix}_{didx:03d}.png"
            self.plotClustering(figname, threshold)
            # クラスタリングの結果から、分類の成功の是非を判定する
            results = self.evaluateClustering(threshold)
            if results[0] != False:
                cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count = results
            else:
                print("Failed to classify: # of cluster was not 2")
                print("{}.".format(current_value))
                failure_param = current_value
                break
                #continue

            # clusteringの際の "isomorphic threshold"
            last_merge = self.Z[-2]
            thresh_obs = last_merge[2]
            # loc_abの値と、分類の成功の是非をリストに格納する
            loc_sim_results.append([didx, current_value, cluster_1_A_purity, cluster_1_B_purity, cluster_2_A_purity, cluster_2_B_purity, cluster_1_count, cluster_2_count, thresh_obs, figname])

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
                failure_param = current_value
                break
            else:
                # 分類に成功する場合のloc_abの値を記録する
                print("Success {}.".format(current_value))
                success_param = current_value

            # 分類の成功の是非を判定する
            if (cluster_1_A_purity >= 0.8 and cluster_1_B_purity <= 0.2 and cluster_2_A_purity <= 0.2 and cluster_2_B_purity >= 0.8) or \
                (cluster_1_A_purity <= 0.2 and cluster_1_B_purity >= 0.8 and cluster_2_A_purity >= 0.8 and cluster_2_B_purity <= 0.2):
                # 分類に成功する場合のloc_abの値を記録する
                success_param = current_value
            else:
                # 分類に失敗する場合のloc_abの値を記録する)
                print("Failure {}.".format(current_value))
                failure_param = current_value
                break
        
        # 仮にsuccess_loc_ab, failure_loc_abの値が定義されていない場合は、それぞれの値をNoneにする
        if "success_param" not in locals():
            success_param = None
        if "failure_param" not in locals():
            failure_param = None
        
        # success_loc_ab, failure_loc_abの値を返す
        return success_param, failure_param, loc_sim_results
    
    # 総合的なシミュレーションを実施できるクラス関数
    # skewed gaussianのパラメータであるloc, scale, alphaを変更して、クラスタリングを行い、その結果を評価する
    # 手順は以下である。
    # 1. 初期の(loc, scale, alpha)はインスタンスを作る時点で設定される
    # 2. AA, AB, BBの3つのデータの分布の条件を変更して、クラスタリングを行う
    # 3. どの分布のどのパラメータ(loc, scale, alpha)を変更するかを指定する
    # 4. 指定する際には例えば、("AA", "loc", min_value, max_value, step)のように指定する
    # 5. 上記の例であれば、AA(self.sets_aa)のlocの値をmin_valueからmax_valueまでstepごとに変更して、クラスタリングを行う
    # 6. 結果の評価は何段階かに分けて行う
    #    変更されたパラメータの値を利用して、クラスタリングを行う。
    #    クラスタリングはAA, AB, BBの３つのskewed gaussianモデルからCCを得て、そのCCを利用してクラスタリングを行う
    #    scipy.cluster.hierarchy.linkageを利用して、ward法でクラスタリングを行う 
    # 7. 1段階目は仮定している”A"と”B”のラベルが指定したしきい値（例：0.6など）で２つのクラスタに分類されるかどうかの判定
    #    この判定は、クラスタリングの結果を見て、クラスタの数が2であるかどうかを判定する(fclusterの戻り値の数が2であるかどうか）
    #    この時点でクラスタの数が2でなければ、分類に失敗したと判定する。その際、失敗した入力条件(loc, scale, alpha)をすべて記録する
    # 8. ２段階目はクラスタの数が2である場合は、クラスタの中の”A”の数がself.n_Aの20%以上であるかどうかを判定する
    #    また、クラスタの中の”B”の数がself.n_Bの20%以上であるかどうかを判定する
    #    この時点で、1つめのクラスタに含まれる”A”の数がself.n_Aの20%以上であるか、もしくは、2つめのクラスタに含まれる”A”の数がself.n_Aの20%以上である場合は、分類に失敗したと判定する
    #    その際、失敗した入力条件(loc, scale, alpha)をすべて記録する
    # 9. 3段階目は2つのクラスタに含まれる、"A", "B"のラベルの数の純度を計算する
    #    例えば、1つめのクラスタに含まれる”A”の数が１つ目のクラスタに含まれる合計のデータ数に対する比率を計算する
    #    1番目のクラスタに含まれるデータ数をn_1, 2番目のクラスタに含まれるデータ数をn_2とする
    #    例えば、purityについては1つ目のクラスタに含まれる”A"のpurityをpurity_1_Aとする n_1_A / n_1 とする
    #    purityは purity_1_A, purity_1_B, purity_2_A, purity_2_B を計算する
    #   この時点で、purity_1_A >= 0.8 and purity_1_B <= 0.2 and purity_2_A <= 0.2 and purity_2_B >= 0.8 または、
    #   purity_1_A <= 0.2 and purity_1_B >= 0.8 and purity_2_A >= 0.8 and purity_2_B <= 0.2 である場合は、分類に成功したと判定する
    # 10. 4段階目は結果を記録する
    #   3段階目で分類に成功した場合に、設定した (loc, scale, alpha) の値をすべて記録する
    #   クラスタリングにより得られた、n_1, n_2, purity_1_A, purity_1_B, purity_2_A, purity_2_B を記録する
    #   さらに 4. で指定した変更の対象としたパラメータの数値について、失敗したクラスタリングの結果、と、最後に成功したクラスタリングの結果の中点をその時の「限界値」(threshold)として記録する
    #   得られた結果については、pandasのDataFrameに格納する
    # 11. 以上の計算について6-10を１サイクルとする
    #   6-10をn_times回繰り返し、結果を記録する。pandasについてはDataFrameをサイクルごとに更新する

# テスト用のmain関数を定義する
if __name__ == "__main__":
    #aa_sets = (-15.2177, 0.9945, 0.0195)
    aa_sets = (-15.2177, 0.99, 0.0195)
    bb_sets = (-8.475, 0.9883, 0.0251)
    ab_sets = (-11.3898, 0.9799, 0.0277)

    shca=SimHCA(100, aa_sets=aa_sets, ab_sets=ab_sets, bb_sets=bb_sets)

    n_cycle = int(sys.argv[1])
    shca.simSmallerDiff("AB", "loc", 0.97, 1.00, 0.003, n_cycle, 0.6)