import AnaCC
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt

class HistSplineRandom:
    def __init__(self, n_datasets):
        print("GOGOGO")
        # 全データセット数
        self.n_datasets = n_datasets
        # A構造のデータセット数
        self.n_A = int(n_datasets/2.0)
        # B構造のデータセット数
        self.n_B = self.n_datasets - self.n_A
        self.isPrepSample=False

        self.ana = AnaCC.AnaCC()


    # self.sample_listという配列にself.n_A, self.n_Bずつラベル"A"と"B"を追加する
    def addSampleName(self):
        self.sample_list = []
        for i in np.arange(0, self.n_A):
            self.sample_list.append("A")
        for j in np.arange(0, self.n_B):
            self.sample_list.append("B")

        self.sample_list = np.array(self.sample_list)
        self.isPrepSample=True

    def makeDistanceMatrix(self, cluster1, cluster2, threshold=0.8):
        import numpy as np
        import CCmodel

        self.addSampleName()
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",len(self.sample_list))
        df_all = self.ana.get_cc_various_values_from_cctable(cluster1, cluster2)
        # AA, BB, AB ごとにモデルを作成してランダム変数を得ることができるようにする
        # まずはAAのモデルを作成する
        # AA
        df_AA = df_all[df_all["cctype"] == "AA"]
        cc_aa = df_AA["cc"].values
        model_AA = CCmodel.CCmodel(cc_aa)
        # AAのデータの数は self.n_A * (self.n_A - 1) / 2
        n_AA = int(self.n_A * (self.n_A - 1) / 2)
        # AAのCCを計算する
        cclist_AA = model_AA.calcCCmulti(n_AA)
        # cclist_AAのヒストグラムを図として保存する
        plt.hist(cclist_AA, bins=10)
        plt.savefig("cclist_AA.png")
        # ccを距離に変換する
        dlist_AA = np.sqrt(1-cclist_AA*cclist_AA)

        # BB
        df_BB = df_all[df_all["cctype"] == "BB"]
        cc_bb = df_BB["cc"].values  
        model_BB = CCmodel.CCmodel(cc_bb)
        # BBのデータの数は self.n_B * (self.n_B - 1) / 2
        n_BB = int(self.n_B * (self.n_B - 1) / 2)
        # BBのCCを計算する
        cclist_BB = model_BB.calcCCmulti(n_BB)
        # ccを距離に変換する
        dlist_BB = np.sqrt(1-cclist_BB*cclist_BB)
        # cclist_BBのヒストグラムを図として保存する
        plt.hist(cclist_BB, bins=10)
        plt.savefig("cclist_BB.png")
        
        # AB
        df_AB = df_all[df_all["cctype"] == "AB"]
        cc_ab = df_AB["cc"].values
        model_AB = CCmodel.CCmodel(cc_ab)
        # ABのデータの数は self.n_A * self.n_B
        n_AB = self.n_A * self.n_B
        # ABのCCを計算する
        cclist_AB = model_AB.calcCCmulti(n_AB)
        # ccを距離に変換する
        dlist_AB = np.sqrt(1-cclist_AB*cclist_AB)
        # cclist_ABのヒストグラムを図として保存する
        plt.hist(cclist_AB, bins=10)
        plt.savefig("cclist_AB.png")

        # n_AA, n_BB, n_AB を表示
        print("n_AA = ", n_AA)
        print("n_BB = ", n_BB)
        print("n_AB = ", n_AB)

        # 距離行列の初期化
        n_all = self.n_datasets
        dist_matrix = np.zeros((2 * n_all, 2 * n_all))

        # A-Aの距離を距離行列に設定
        index = 0
        for i in range(self.n_A):
            for j in range(i+1, self.n_A):
                dist_matrix[i][j] = dlist_AA[index]
                dist_matrix[j][i] = dlist_AA[index]
                index += 1

        # B-Bの距離を距離行列に設定
        index = 0
        for i in range(self.n_B):
            for j in range(i+1, self.n_B):
                dist_matrix[i+self.n_B][j+self.n_B] = dlist_BB[index]
                dist_matrix[j+self.n_B][i+self.n_B] = dlist_BB[index]
                index += 1

        # A-Bの距離を距離行列に設定
        index = 0
        for i in range(self.n_A):
            for j in range(self.n_A):
                dist_matrix[i][j+self.n_A] = dlist_AB[index]
                dist_matrix[j+self.n_A][i] = dlist_AB[index]
                index += 1
        
        # squareform を使用して距離行列をベクトルに変換する
        from scipy.spatial.distance import squareform 
        dist_matrix = squareform(dist_matrix)

        # dist_matrixを表示
        print(dist_matrix)
        # dist_matrixのdimensionを表示
        print(dist_matrix.shape)
        print(type(dist_matrix))
        # dist_matrixを1次元配列に変換
        dist_matrix = dist_matrix.tolist()
        sample_list = self.sample_list.tolist()

        # linkage関数を使用して階層的クラスタリングを行う
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(dist_matrix, 'ward')

        # dendrogram関数を使用してデンドログラムを作成する
        #dendrogram(Z)
        sample_list = ["A" if i < self.n_A else "B" for i in range(self.n_datasets)]
        s = np.array(sample_list)
        #dendrogram(Z, labels=s,leaf_font_size=10)
        dendrogram(Z, leaf_font_size=10)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        plt.show()

    def makeCC(self, cluster1, cluster2):
        from scipy.stats import skewnorm
        import numpy as np
        import CCmodel

        df_all = self.ana.get_cc_various_values_from_cctable(cluster1, cluster2)
        # AA, BB, AB ごとにモデルを作成してランダム変数を得ることができるようにする
        # まずはAAのモデルを作成する
        # AA
        df_AA = df_all[df_all["cctype"] == "AA"]
        cc_aa = df_AA["cc"].values
        self.model_AA = CCmodel.CCmodel(cc_aa)
        # BB
        df_BB = df_all[df_all["cctype"] == "BB"]
        cc_bb = df_BB["cc"].values  
        self.model_BB = CCmodel.CCmodel(cc_bb)
        # AB
        df_AB = df_all[df_all["cctype"] == "AB"]
        cc_ab = df_AB["cc"].values
        self.model_AB = CCmodel.CCmodel(cc_ab)

        if self.isPrepSample==False:
            self.addSampleName()

        # aa_sets, bb_sets, ab_sets にはそれぞれ(alpha, loc, scale)のタプルが格納されている
        # この関数内で利用できるように展開する
        self.ccAA = []
        self.ccAB = []
        self.ccBB = []
        self.distance_matrix = []

        aa_count=0
        bb_count=0
        ab_count=0
        for idx1,s1 in enumerate(self.sample_list):
            for s2 in self.sample_list[idx1+1:]:
                if s1=="A" and s2=="A":
                    cctmp = self.model_AA.rand_func(0)
                    self.ccAA.append(cctmp)
                    aa_count+=1
                elif s1=="B" and s2=="B":
                    cctmp = self.model_BB.rand_func(0)
                    self.ccBB.append(cctmp)
                    bb_count+=1
                else:
                    cctmp = self.model_AB.rand_func(0)
                    self.ccAB.append(cctmp)
                    ab_count+=1

                if cctmp>1.0:
                    cctmp=1.0

                # distance = sqrt(1-cc^2)
                distance = np.sqrt(1-cctmp*cctmp)
                self.distance_matrix.append(distance)

        import pandas as pd
        # AA のDataframe
        df_AA = pd.DataFrame(self.ccAA)
        df_AA.to_csv("ccAA.csv")
        # BB のDataframe
        df_BB = pd.DataFrame(self.ccBB)
        df_BB.to_csv("ccBB.csv")
        # AB のDataframe
        df_AB = pd.DataFrame(self.ccAB)
        df_AB.to_csv("ccAB.csv")

        # distance_matrixをCSVにしておく
        df = pd.DataFrame(self.distance_matrix)
        df.to_csv("distance_matrix.csv")

    def tttt(self, ccAA, ccAB, ccBB):
        import numpy as np
        from scipy.cluster.hierarchy import linkage, dendrogram

        # AとBのデータを生成する
        n_all = 50

        # 距離行列を初期化するために、(2*n_data)x(2*n_data)のゼロ行列を作成する
        dist_matrix = np.zeros((2*n_data, 2*n_data))

        # A-Aの相関係数を使用して、距離行列の対角線部分を埋める
        for i in range(n_data):
            for j in range(i+1, n_data):
                if i == j:
                    continue
                dist_matrix[i,j] = np.sqrt(2*(1-cclist_AA[i*(2*n_data-i-1)//2+j-i-1]))
                dist_matrix[j,i] = dist_matrix[i,j]

        # B-Bの相関係数を使用して、距離行列の対角線部分を埋める
        for i in range(n_data, 2*n_data):
            for j in range(i+1, 2*n_data):
                if i == j:
                    continue
                dist_matrix[i,j] = np.sqrt(2*(1-cclist_BB[i*(2*n_data-i-1)//2+j-i-1]))
                dist_matrix[j,i] = dist_matrix[i,j]

        # A-Bの相関係数を使用して、距離行列の非対角線部分を埋める
        for i in range(n_data):
            for j in range(n_data, 2*n_data):
                dist_matrix[i,j] = np.sqrt(2*(1-cclist_AB[i*(n_data)+(j-n_data)]))
                dist_matrix[j,i] = dist_matrix[i,j]

        # linkage関数を使用して階層的クラスタリングを行う
        Z = linkage(dist_matrix, 'ward')

        # dendrogram関数を使用してデンドログラムを作成する
        dendrogram(Z)


    def proc(self, cluster1, cluster2, threshold=0.80):
        error_flag = self.ana.crossCheck(cluster1, cluster2, cc_thresh=0.9165, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst")
        if error_flag:
            print("Error: cross check failed")
            sys.exit(1)

        print("##################################################")
        self.makeCC(cluster1, cluster2)
        print("##################################################")

        # HCA linkage
        from scipy.cluster.hierarchy import linkage
        z = linkage(self.distance_matrix, method="ward")
        # import dendrogram
        from scipy.cluster.hierarchy import dendrogram
        self.dendrogram = dendrogram(z, labels=self.sample_list,leaf_font_size=8)
        #デンドログラムを表示する
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        plt.savefig("dendrogram.png")
        plt.show()
            
# mainが定義されていなかったら
if __name__ == "__main__":
    import sys
    ana = AnaCC.AnaCC()
    print("From cluster number")
    cluster1=sys.argv[1]
    cluster2=sys.argv[2]
    # しきい値を引数から得る
    # しきい値が指定されていなければ0.8を使う
    if len(sys.argv) > 3:
        threshold = float(sys.argv[3])
    else:
        threshold = 0.9165

    hsr=HistSplineRandom(100)
    hsr.proc(cluster1, cluster2, threshold=0.8)
    #hsr.makeDistanceMatrix(cluster1, cluster2, threshold=0.8)
