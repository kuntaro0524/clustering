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

    # 分岐点のWard距離の数値を表示するための関数
    def add_distance_labels(self, ax, linkage_matrix, threshold):
        for i, d, c in zip(linkage_matrix[:, 0], linkage_matrix[:, 2], linkage_matrix[:, 3]):
            x = 0.5 * (self.dendrogram['icoord'][i][1] + self.dendrogram['icoord'][i][2])
            y = d
            if y > threshold:
                ax.text(x, y, f'{y:.2f}', color='r')
            else:
                ax.text(x, y, f'{y:.2f}', color='k')

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

        # 続きでやりたいこと
        # fclusterはある閾値で線を引いたときにクラスタごとにデータにインデクスをつけてくれる
        # 例）[1,1,1, 2,2,2,3,3,3,4,4,4]
        # なので、それぞれのグループの中で、最もWard distanceが大きいときだけデンドログラム上にward distanceを表示する
        # ということができれば良い

        print("CLUCLUCS")
        print(len(cluster_list),cluster_list)
        print("CLUCLUCS")

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
        # isomorphic threshold以下のクラスタは色を変える
        # また、isomorphic threshold以下のクラスタをまとめる
        # その際に、まとめたクラスタのデータの個数をカウントする
        # さらに、まとめたクラスタのデータの名前をログファイルに書き出す
        plt.figure(figsize=(10, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        self.dendrogram = hierarchy.dendrogram(self.Z, labels=self.name_list2, leaf_font_size=8, color_threshold=self.isomorphic_thresh)
        # 各ノードのところに、ward distanceを表示する
        # self.add_distance_labels(ax, self.Z, threshold=self.isomorphic_thresh)
        plt.axhline(y=self.isomorphic_thresh, color='r', linestyle='--')

        print("JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJjj")
        for idx,c in enumerate(self.dendrogram['icoord']):
            # idx はクラスタのインデックスとなる→これがKAMOのインデックスとほぼ同じなのではないか
            # このidxを使ってこのクラスタのward distanceを取得する
            wd = self.dendrogram['dcoord'][idx][2]
            print(c,wd)
            # c[1]とc[2]の中点をX座標、wdをY座標としてテキストを描画する
            # テキストの内容 "%5.1f:%d" % (wd, idx)
            x = 0.5 * (c[1] + c[2])
            y = wd
            plt.text(x, y, "%8.3f : %5d" % (wd, idx), color='k')
        plt.savefig("dendrogram.png")
        plt.show()
        print("JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJjj")
         
if __name__ == "__main__":
    cctable = sys.argv[1]
    filename_list = sys.argv[2]
    cc2dendrogram = CC2Dendrogram(cctable, filename_list)
    cc2dendrogram.run()