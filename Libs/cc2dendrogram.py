import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

class CC2Dendrogram:
    def __init__(self, cctable, filename_list):
        self.cctable = cctable
        self.filename_list = filename_list

        # Ratio
        self.it_ratio = 0.7

    def run(self):
        with open(self.cctable) as f:
            lines = f.readlines()
        cc_list = []
        dis_list = []

        for line in lines[1:]:
            x = line.split()
            cc_list.append(float(x[2]))
            dis_list.append(np.sqrt(1 - float(x[2]) ** 2))

        with open(self.filename_list) as f2:
            lines = f2.readlines()

        name_list = []

        for line in lines:
            name_list.append(line.strip())

        Z = hierarchy.linkage(dis_list, 'ward')

        # Zの情報をファイルに書き出す "z.txt"
        ofile = open("z.txt", "w")
        for i in range(Z.shape[0]):
            ofile.write(str(Z[i]) + "\n")
        ofile.close()

        last_merge = Z[-1]
        thresh0 = last_merge[2]

        last_merge = Z[-2]
        threshold = last_merge[2]

        max_thresh = threshold
        print("max_thresh", thresh0)
        new_thresh = threshold / thresh0

        # isomorphic thresholdを計算する (ward distance)
        isomorphic_thresh = self.it_ratio * max_thresh
        print("isomorphic_thresh=", isomorphic_thresh)

        # isomorphic thresholdよりも小さいWard distanceを持つクラスタを抽出する
        # それらのクラスタを構成するデータの個数をカウントする
        # ログファイルは 'cluster.txt'とする
        # またログファイルに、それぞれのクラスタのノードを構成するデータの名前を書き出す
        # データの名前というのは、self.filename_listに記載されているものとする
        # 階層的クラスタリングの結果はデータ数が少ないものから結合されているので、
        # Zのward distanceの数値が大きいものから順に、isomorphic thresholdよりも小さいものを探す
        # isomorphic thresholdより小さなward distanceを持つクラスタが見つかったら、
        # そのクラスタを構成するデータの個数をカウントする
        # そのクラスタを構成するデータの名前をログファイルに書き出す
        # このためには再帰的にZの内容を調べる必要がある
        # そのためには、Zの内容を調べる関数を作成する必要がある
        # その関数の名前は、search_Z()とする
        # search_Z()の引数は、Z, isomorphic_thresh, name_list, log_fileとする
        # Zは、階層的クラスタリングの結果である
        # isomorphic_threshは、isomorphic thresholdである
        # name_listは、データの名前のリストである 
        # log_fileは、ログファイルの名前である
        # search_Z()は、isomorphic thresholdよりも小さいward distanceを持つクラスタを見つける
        # そのクラスタを構成するデータの個数をカウントする
        # そのクラスタを構成するデータの名前をログファイルに書き出す

        def search_Z(Z, isomorphic_thresh, name_list, log_file):
            # Zの内容を調べる
            # Zの各行の内容は、[クラスタ1, クラスタ2, ward distance, クラスタを構成するデータの個数]である
            # クラスタ1とクラスタ2は、クラスタを構成するデータの名前のインデックスである
            # ファイル名を取得する場合には self.name_list[クラスタ1]とすれば良い（以下同文）
            # クラスタを構成するデータの個数が2のときには、クラスタを構成するデータの名前をログファイルに書き出す
            # 2より大きいときには、クラスタ１のインデックスについて、再帰的にZの内容を調べる
            # クラスタ2のインデックスについても、再帰的にZの内容を調べる
         
        plt.figure()
        dn = hierarchy.dendrogram(Z, labels=name_list)
        plt.savefig("dendrogram.jpg")
        plt.show()

if __name__ == "__main__":
    cctable = sys.argv[1]
    filename_list = sys.argv[2]
    cc2dendrogram = CC2Dendrogram(cctable, filename_list)
    cc2dendrogram.run()