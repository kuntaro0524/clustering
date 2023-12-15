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

        self.name_list = []

        for line in lines:
            self.name_list.append(line.strip())

        self.Z = hierarchy.linkage(dis_list, 'ward')

        # Zの情報をファイルに書き出す "z.txt"
        ofile = open("z.txt", "w")
        for i in range(self.Z.shape[0]):
            ofile.write(str(self.Z[i]) + "\n")
        ofile.close()

        last_merge = self.Z[-1]
        thresh0 = last_merge[2]

        last_merge = self.Z[-2]
        threshold = last_merge[2]
        # max_threshは、ward distanceの最大値である
        max_thresh = threshold
        print("max_thresh", thresh0)
        # isomorphic thresholdを計算する (ward distance)
        self.isomorphic_thresh = self.it_ratio * max_thresh
        print("isomorphic_thresh=", self.isomorphic_thresh)

        # isomorphic thresholdよりも小さいWard distanceを持つクラスタを抽出する
        # しかしすべてを抽出すると、クラスタ数が多すぎるので上位ノードのみを抽出する
        # ここでいう上位、というのはWard distanceがより大きいものと定義する
        # 最終的には、Ward distanceのしきい値を設定し、それを下回るクラスタを抽出するが、
        # しきい値の下のクラスタのうち、最もWard distanceが大きいもののみを抽出する
        # ここでいう抽出は、ノードを構成するデータの個数をカウントし、さらに、
        # ノードを構成するデータの名前をログファイルに書き出すことと定義する
        # ここでいうデータの名前とは、self.name_listに記載されているものとする
        # ログファイルは 'cluster.txt'とする。
        # self.Zを解析し、上記のように結果をまとめる。

        # <Zの内容>
        # [51.         73.          0.10846746  2.        ]
        # [15.         17.          0.11556431  2.        ]
        # [ 2.         90.          0.11740975  4.        ]
        # [27.         88.          0.12078673  5.        ]
        # </Zの内容>

        # Zの内容は、階層的クラスタリングの結果である
        # Zの各行の内容は、[クラスタ1, クラスタ2, ward distance, クラスタを構成するデータの個数(以下：構成データ数と呼ぶ)]である
        # クラスタ1とクラスタ2は、クラスタを構成するデータの名前のインデックスに対応している
        # このインデックスを c1_index, c2_indexとする
        # ファイル名を取得する場合には self.name_list[クラスタ1]とすれば良い（以下同文）

        # 通常、name_listのデータ数をnとすると、cc_tableのデータ数はn(n-1)/2となる
        # まずZのインデックスがlen(name_list)-1 より大きい場合には、name_listから名前を取得できない
        # そのため、Zのインデックスがlen(name_list)-1 より大きい場合には、
        # Z[c1_index - len(name_list)]もしくはZ[c2_index - len(name_list)]を参照する
        # これを繰り返すことにより、すべてのインデックスをname_listに含まれるファイル名に変換することができる

        # 例えば、name_listのデータ数が10である場合には、
        # Zのインデックスが10以上の場合には、Zのインデックスから10を引く
        # このインデックスがname_listのデータ数よりも大きい場合には、再び10を引く必要がある
        # このようにして、name_listのデータ数を超えないインデックスを得ることができる

        # まずZの内容についてWard distanceの値が大きい順にソートする
        # self.Zについて、Ward distanceの値が大きい順にソートする
        # ソートした結果は、self.Z_sortedとする
        self.Z_sorted = self.Z[self.Z[:,2].argsort()[::-1]]

        # ソートした結果をファイルに書き出す "z_sorted.txt"
        ofile = open("z_sorted.txt", "w")
        for i in range(self.Z_sorted.shape[0]):
            ofile.write(str(self.Z_sorted[i]) + "\n")
        ofile.close()

        # ソートした結果の最初のクラスタはすべてのデータを含んでいるはずなので、
        # そのクラスタを構成するデータの個数をカウントする
        zenma = self.Z_sorted[0]
        c1_index = int(zenma[0])
        c2_index = int(zenma[1])
        ward_distance = zenma[2]
        n_data = int(zenma[3])
        print(c1_index, c2_index, ward_distance, n_data)
        print(len(self.name_list))

        # c1_index, c2_indexがname_listのデータ数よりも大きい場合には、
        # c1_index, c2_indexからname_listのデータ数を引いて直下のクラスタを調査する
        # c1_indexについてトレースしていく
        # インデックスを与えたときにそのインデックスを解析する関数を作成する
        def index2name(index):
            flist = []
            indlist = []
            # print("processing index:", index)
            if index < len(self.name_list):
                # print("reached to the file list: index=", index)
                flist.append(self.name_list[index])
                indlist.append(index)
                return flist, indlist
            else:
                # print("Moving to the child nodes...")
                zdata_tmp = self.Z[index - len(self.name_list)]
                # print("Next Z:", zdata_tmp)
                c1_index = int(zdata_tmp[0])
                c2_index = int(zdata_tmp[1])
                c1_name, c1_ind = index2name(c1_index)
                c2_name, c2_ind = index2name(c2_index)
                flist.extend(c1_name)
                flist.extend(c2_name)
                indlist.extend(c1_ind)
                indlist.extend(c2_ind)
                return flist, indlist

        # Convert the returned file name list into a one-dimensional array and concatenate them
        print(">>>>>>>>>>>>>>>>>")
        f,ff=index2name(c1_index)
        print(len(ff))
        # print(f)
        print(ff)
        f,ff=index2name(c2_index)
        print(ff)

        # fcluster関数を利用してしきい値よりも小さなクラスタのデータをまとめる
        # しきい値は、isomorphic thresholdとする
        cluster_list = hierarchy.fcluster(self.Z, self.isomorphic_thresh, criterion='distance')
        print(len(cluster_list),cluster_list)

        # cluster_listは、データの個数と同じ長さのリストである
        # cluster_listの各要素は、データのインデックスに対応している
        # インデックスが重複しているものをまとめる
        cluster_list2 = list(set(cluster_list))
        print(cluster_list2)

        # cluster_list2の各養素はself.name_listのインデックスに対応している
        # cluster_list2 の要素数は self.name_listと同じである
        for name_index,cluster_num in enumerate(cluster_list):
            print(name_index, cluster_num)

        # self.name_listを少し簡単な名前にする
        # 上から順に、0, 1, 2, 3, ... とする
        # このリストをself.name_list2とする
        self.name_list2 = []
        for i in range(len(self.name_list)):
            self.name_list2.append(str(i))

        # デンドログラムの描画
        # self.Zを使ってデンドログラムを描画する
        # その際に、self.name_listをラベルにする
        # また、isomorphic thresholdを赤い線で描画する
        plt.figure(figsize=(10, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        hierarchy.dendrogram(
            self.Z,
            labels=self.name_list2,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        plt.axhline(y=self.isomorphic_thresh, color='r', linestyle='-')
        plt.show()  


if __name__ == "__main__":
    cctable = sys.argv[1]
    filename_list = sys.argv[2]
    cc2dendrogram = CC2Dendrogram(cctable, filename_list)
    cc2dendrogram.run()