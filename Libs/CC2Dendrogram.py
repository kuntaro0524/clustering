import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import pandas as pd

class CC2Dendrogram:
    def __init__(self, cctable, filename_list):
        self.cctable = cctable
        self.filename_list = filename_list
        self.isRead = False

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

    def checkDist(self):
        # self.dis_listの中で反射数が３未満になっているものを探す
        for i in range(len(self.dis_list)):
            if self.dis_list[i] < 0.001:
                print("Found! index=",i, " Ward distance=", self.dis_list[i])
                print(self.Z[i])

    def cleanData(self):
        # dis_listがnumpy配列であると仮定
        # 無限大の要素をチェック
        inf_indices = np.where(np.isinf(self.dis_list))
        print("無限大の要素の位置：", inf_indices)

        # NaNの要素をチェック
        nan_indices = np.where(np.isnan(self.dis_list))
        print("NaNの要素の位置：", nan_indices)

        # 元の self.dis_listの長さ
        print("元のself.dis_listの長さ：", len(self.dis_list))

        # dis_listがnumpy配列であると仮定
        # 無限大やNaNを含むかどうかをチェック
        if not np.all(np.isfinite(self.dis_list)):
            print("replace inf and nan to 0")
            # 無限大やNaNを別の値で置き換える（例えば0や平均値など）
            self.dis_list = np.nan_to_num(self.dis_list)

        print("最終のself.dis_listの長さ：", len(self.dis_list))
        
    def rejectResort(self):
        # cctable.dat を読み込む
        data = pd.read_csv('cctable.dat', delim_whitespace=True)

        # nrefl が3未満の行を特定し、関連するインデックスを取得
        exclude_indices = set()
        for index, row in data.iterrows():
            if row['nref'] < 3:
                exclude_indices.add(row['i'])
                exclude_indices.add(row['j'])

        print(len(exclude_indices), exclude_indices)

        # 関連するインデックスを除外
        filtered_data = data[~data['i'].isin(exclude_indices) & ~data['j'].isin(exclude_indices)]
        
        # 除外されたデータを表示
        print("Rejected datasets")
        print(data[data['i'].isin(exclude_indices) | data['j'].isin(exclude_indices)])
        # 除外されたデータの数を表示
        print("# of Rejected datasets")
        print(len(data[data['i'].isin(exclude_indices) | data['j'].isin(exclude_indices)]))

        # 新しいインデックスを作成
        unique_indices = sorted(set(filtered_data['i']).union(set(filtered_data['j'])))
        index_map = {old_index: new_index for new_index, old_index in enumerate(unique_indices)}

        # 新しいインデックスでデータを更新
        filtered_data['i'] = filtered_data['i'].map(index_map)
        filtered_data['j'] = filtered_data['j'].map(index_map)

        # 相関係数の距離行列を作成
        size = len(unique_indices)
        distance_matrix = squareform([1 - cc for cc in filtered_data['cc']])
        linkage_matrix = linkage(distance_matrix, method='ward')

        # 結果を表示
        print(linkage_matrix)
        # self.dis_list には
        print("##########################")
        print(self.dis_list)
        print("##########################")

    # end of rejectResort

    def readData2(self):
        # cctable.dat を読み込む pandas.DataFrameとして読み込む
        # 1行目はヘッダーなのでカラム名として利用する
        # columns は ['i', 'j', 'cc', 'nref'] となる
        # columns のデータ型は integer, integer, float, integer である
        # Dataframeにはそのデータ型で格納する
        maindf = pd.read_csv('cctable.dat', delim_whitespace=True, dtype={'i':int, 'j':int, 'cc':float, 'nref':int})
        # cc の数値がnanの行を特定し、関連するインデックスを取得
        tmpdf = maindf[maindf['cc'].isnull()]
        # 悪い結果を出しているデータのインデクスを格納する辞書
        idx_bad = {}
        # nans 配列を準備して、nanのインデックスを格納する
        nans = []
        for index, row in tmpdf.iterrows():
            tmpi= int(row['i'])
            tmpj= int(row['j'])
            nans.append((tmpi,tmpj))
            idx_bad[tmpi] = idx_bad.get(tmpi, 0) + 1
            idx_bad[tmpj] = idx_bad.get(tmpj, 0) + 1
        
        # 次に、nrefl が3未満の行を特定し、関連するインデックスを取得しnansに追加する
        lack_refdf = maindf[maindf['nref'] < 3]
        for index, row in lack_refdf.iterrows():
            tmpi= int(row['i'])
            tmpj= int(row['j'])
            # すでにnansに入っている場合には、追加しない
            if (tmpi, tmpj) in nans: 
                print("あるよ！")
                continue
            nans.append((tmpi,tmpj))
            idx_bad[tmpi] = idx_bad.get(tmpi, 0) + 1
            idx_bad[tmpj] = idx_bad.get(tmpj, 0) + 1

        # idx_badをリストに変換してからsortする
        idx_bad_list = list(idx_bad.items())
        idx_bad_list.sort(key=lambda x:x[1])

        # 削除するデータのインデックス
        remove_idxes = set()

        for idx, badcount in reversed(idx_bad_list):
            print("Current nans: ", nans)
            print("processing idx: ", idx)
            if len([x for x in nans if idx in x]) == 0: continue
            print("GOGOGOGOG")
            remove_idxes.add(idx)

            nans = [x for x in nans if idx not in x]
            if len(nans) == 0: break

        # remove_idxes の数
        print("remove_idxes: ", len(remove_idxes))
        
        # 利用するインデックス
        use_idxes = [x for x in range(len(maindf)) if x not in remove_idxes]
        print(len(use_idxes))

    def readData(self):
        with open(self.cctable) as f:
            lines = f.readlines()
        self.cc_list = []
        self.dis_list = []
        self.nref_list = []

        self.bad_file_indices = []
        for line in lines[1:]:
            x = line.split()
            self.cc_list.append(float(x[2]))
            self.dis_list.append(np.sqrt(1 - float(x[2]) ** 2))
            self.nref_list.append(int(x[3]))
            # nrefl check
            if int(x[3]) < 3:
                # print("number of reflection is less than 3")
                # データのインデックスを保存する
                self.bad_file_indices.append(int(x[0]))
                self.bad_file_indices.append(int(x[1]))
            else:
                self.cc_list.append(float(x[2]))
                self.dis_list.append(np.sqrt(1 - float(x[2]) ** 2))

        # self.bad_file_indicesの重複をなくす
        self.bad_file_indices = list(set(self.bad_file_indices))
        print("Bad files:", self.bad_file_indices)

        with open(self.filename_list) as f2:
            lines = f2.readlines()
        
        self.name_list = []

        for line in lines:
            self.name_list.append(line.strip())

        # self.bad_file_indecesのインデックスの要素を self.filename_listから削除する
        # インデックスがindices_to_removeに含まれていない要素だけを新しいリストに含める
        filtered_list = [item for idx, item in enumerate(self.name_list) if idx not in self.bad_file_indices]
        print(len(self.name_list))
        print(len(filtered_list))

        self.isRead = True

    def writeZinfo(self, prefix):
        ofile = open(prefix + "_z.txt", "w")
        # 書き出す self.Zの要素は、[c1_index, c2_index, ward_distance, n_data] である
        # c1_index, c2_index は四桁の整数として書き出す
        # ward_distance は小数点以下3桁まで書き出す
        # n_data は整数として書き出す
        for i in range(self.Z.shape[0]):
            ofile.write("%04d %04d %5.3f %d\n"%(self.Z[i][0], self.Z[i][1], self.Z[i][2], self.Z[i][3]))
        ofile.close()
        # sorted
        ofile = open(prefix + "_z_sorted.txt", "w")
        for i in range(self.Z_sorted.shape[0]):
            ofile.write("%04d %04d %5.3f %d\n"%(self.Z_sorted[i][0], self.Z_sorted[i][1], self.Z_sorted[i][2], self.Z_sorted[i][3]))
        ofile.close()

    def prepZ(self):
        # Z を計算する
        if self.isRead == False:
            self.readData()

        # cleanData
        self.cleanData()

        self.Z = hierarchy.linkage(self.dis_list, 'ward')

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

        # Z, Z_sorted をファイルに書き出す
        self.writeZinfo("cc")

    def readKAMOCClist(self, filename="CLUSTERS.txt"):
        # CLUSTERS.txtを読み込む
        # pandas dataframeとして読む
        # １列目はクラスタID, 2列目はクラスタのデータ数, 3列目はクラスタのWard距離, 4列目以降はクラスタを構成するデータのインデックス
        # ただし、４列目以降の数値は整数が複数入っているため、４列目以降の整数をリストにしてから、pandas dataframeの
        # data_indices カラムに保存する
        lines = open(filename).readlines()
        cluster_id_list = []
        cluster_n_data_list = []
        cluster_ward_distance_list = []
        cluster_data_indices_list = []
        for line in lines[1:]:
            x = line.split()
            cluster_id_list.append(int(x[0]))
            cluster_n_data_list.append(int(x[1]))
            cluster_ward_distance_list.append(float(x[2]))
            cluster_data_indices_list.append([int(i) for i in x[3:]])
        # pandas dataframeを作成する
        df = pd.DataFrame()
        df['cluster_id'] = cluster_id_list
        df['n_data'] = cluster_n_data_list
        df['ward_distance'] = cluster_ward_distance_list
        df['data_indices'] = cluster_data_indices_list
        # 確認のために data_indicesの要素数をn_dataとともに表示
        for i in range(df.shape[0]):
            print(df['n_data'][i], len(df['data_indices'][i]))

        return df

    def divideClusterWithIsomorphicThreshold(self):
        # CC clustering information from "CLUSTERS.txt"
        df = self.readKAMOCClist()
        # df の中で、最大のWard距離を抽出する
        max_ward_distance = df['ward_distance'].max()
        # isomorphic thresholdを計算する
        self.isomorphic_threshold = self.it_ratio * max_ward_distance
        print("isomorphic_threshold=", self.isomorphic_threshold)
        # isomorphic thresholdよりも大きいWard距離を持つクラスタを抽出する
        df2 = df[df['ward_distance'] >= self.isomorphic_threshold]
        print(df2)

    def index2name(self, index):
        # c1_index, c2_indexがname_listのデータ数よりも大きい場合には、
        # c1_index, c2_indexからname_listのデータ数を引いて直下のクラスタを調査する
        # c1_indexについてトレースしていく
        # インデックスを与えたときにそのインデックスを解析する関数を作成する
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
            c1_name, c1_ind = self.index2name(c1_index)
            c2_name, c2_ind = self.index2name(c2_index)
            flist.extend(c1_name)
            flist.extend(c2_name)
            indlist.extend(c1_ind)
            indlist.extend(c2_ind)
            return flist, indlist
        
    def run(self):
        if self.isRead == False:
            self.readData()
        self.prepZ()
        # ソートした結果の最初のクラスタはすべてのデータを含んでいるはずなので、
        # そのクラスタを構成するデータの個数をカウントする
        zenma = self.Z_sorted[0]
        c1_index = int(zenma[0])
        c2_index = int(zenma[1])
        ward_distance = zenma[2]
        n_data = int(zenma[3])
        # print(c1_index, c2_index, ward_distance, n_data)
        # print(len(self.name_list))

        # Convert the returned file name list into a one-dimensional array and concatenate them
        print(">>>>>>>>>>>>>>>>>")
        f,ff=self.index2name(c1_index)
        print(len(ff))
        print(f)
        print(ff)
        f,ff=self.index2name(c2_index)
        print(ff)

        # fcluster関数を利用してしきい値よりも小さなクラスタのデータをまとめる
        # しきい値は、isomorphic thresholdとする
        cluster_list = hierarchy.fcluster(self.Z, self.isomorphic_thresh, criterion='distance')
        print("==== CLUSTER_LIST =====")
        print(len(cluster_list),cluster_list)
        print("==== CLUSTER_LIST =====")

        # cluster_list2の各養素はself.name_listのインデックスに対応している
        # cluster_list2 の要素数は self.name_listと同じである
        # self.name_listを少し簡単な名前にする
        # 上から順に、0, 1, 2, 3, ... とする
        # このリストをself.name_list2とする
        self.name_list2 = []
        for i in range(len(self.name_list)):
            self.name_list2.append(str(i))

        # 続きでやりたいこと
        # fclusterはある閾値で線を引いたときにクラスタごとにデータにインデクスをつけてくれる
        # 例）cluster_list = [1,2,1, 3,2,3,3,1,3,4,4,4]
        # この場合には、1,2,3,4の4つのクラスタがあることになる
        # このクラスタのインデクスを使って最大のWard距離を持つクラスタを探す
        # まずcluster_listの要素とクラスタの情報をリンクさせる
        # cluster_list[i] = j とすると self.Z[i][2]のWard距離を取得することができる
        # 同一のjを持つ要素のうち、Ward距離が最大のものを探す
        # まず、cluster_listについて何種類あるかを調べる
        self.dendrogram = hierarchy.dendrogram(self.Z, labels=self.name_list2, leaf_font_size=8, color_threshold=self.isomorphic_thresh)
        individual_cluster = set(cluster_list)
        n_groups = len(individual_cluster)

        values = self.findThresholdCluster()
        print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
        for value in values:
            print(value)
            print(self.Z[value[0]])
        print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")

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

        # 特定のノードに Ward distance と cluster number を表示する
        # values に入っているものがそれに相当する
        # values には (index, Z[index]) が入っている
        # index は self.Z のインデックスである
        # Z[index] は、[c1_index, c2_index, ward_distance, n_data] である
        # ward_distanceはこの数値を利用する
        # 数値を書き込む座標は dendrogram['icoord'], dendrogram['dcoord'] を利用する
        # icoord は、各ノードのX座標のリストである (4つの要素を持つリスト)
        # dcoord は、各ノードのY座標のリストである (4つの要素を持つリスト)
        for idx,c in enumerate(values):
            # index_label & word distance
            label = "%03d\n(%5.3f)"%(c[0], c[1][2])
            # デンドログラム上の座標を取得する
            # dcoord は、各ノードのY座標のリストであり、Ward distanceでもある
            # このY座標を利用し、 c[1][2] と比較、差分が0.001以下の場合には
            # X座標を取得する
            xcode = 0
            ycode = 0
            for idx2,d in enumerate(self.dendrogram['dcoord']):
                if abs(d[1] - c[1][2]) < 0.001:
                    xcode = self.dendrogram['icoord'][idx2][1]
                    ycode = d[1]
                    break
            # label を描画する
            plt.text(xcode, ycode, label, color='k')

        # self.isomorphic_thresh を描画する
        plt.axhline(y=self.isomorphic_thresh, color='r', linestyle='--')
        plt.show()

    def drawDendrogram(self):
        # Z を計算する
        if self.isRead == False:
            self.readData()
        self.cleanData()
        self.Z = hierarchy.linkage(self.dis_list, 'ward')
        #シンプルにデンドログラムを描く
        plt.figure(figsize=(50, 50))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        self.dendrogram = hierarchy.dendrogram(self.Z, labels=self.name_list, leaf_font_size=8, color_threshold=0.7)
        # dcoordを利用して self.Z のデータインデックスを表示する
        # dcoord は、各ノードのY座標のリストである
        # このY座標を利用し、 self.Z のデータインデックスと比較、差分が0.001以下の場合には
        # X座標を取得する
        for idx,c in enumerate(self.Z):
            # index_label & word distance
            label = "%03d"%(idx)
            # デンドログラム上の座標を取得する
            # dcoord は、各ノードのY座標のリストであり、Ward distanceでもある
            # このY座標を利用し、 c[1][2] と比較、差分が0.001以下の場合には
            # X座標を取得する
            xcode = 0
            ycode = 0
            for idx2,d in enumerate(self.dendrogram['dcoord']):
                # print(d[1], c[2])
                if abs(d[1] - c[2]) < 0.001:
                    xcode = self.dendrogram['icoord'][idx2][1]
                    ycode = d[1]
                    break
            # label を描画する
            plt.text(xcode, ycode, label, color='k')
        plt.savefig("dendrogram.png")
        plt.show()

    # data_indexは self.Zのインデックスを示している
    def findJustBeforeThresholdStepWiseFromLowerValue(self, data_index):
        # self.Zの中にdata_indexを含まれている要素を探す
        # その時のZのインデックスも取得する
        # そのときのward distanceを表示する
        results_list = []
        for i in range(self.Z.shape[0]):
            c1_index = int(self.Z[i][0])
            c2_index = int(self.Z[i][1])
            if c1_index == data_index or c2_index == data_index:
                print("Found! index=",i, " Ward distance=", self.Z[i][2])
                print(self.Z[i])
                results_list.append((i,self.Z[i]))
                tmp_ward = self.Z[i][2]
                if tmp_ward > self.isomorphic_thresh:
                    print("Found! But the ward distance is larger than the isomorphic threshold")
                    return []
                else:
                    next_index = i + len(self.name_list)
                    print("Next index=", next_index)
                    tmp_list = self.findJustBeforeThresholdStepWiseFromLowerValue(next_index)
                    results_list.extend(tmp_list)
                    return results_list

    def findThresholdCluster(self):
        # fclusterを利用してしきい値よりも小さなクラスタのデータをまとめる
        # しきい値は、isomorphic thresholdとする
        cluster_list = hierarchy.fcluster(self.Z, self.isomorphic_thresh, criterion='distance')
        # cluster_listの要素が同じものを辞書にまとめる
        cluster_dict = {}
        for i,cluster_num in enumerate(cluster_list):
            if cluster_num not in cluster_dict:
                cluster_dict[cluster_num] = []
            cluster_dict[cluster_num].append(i)

        # cluster_dictの中でcluster_numごとに一つだけ要素を取り出す
        # その要素をcluster_list2とする
        cluster_list2 = []
        for cluster_num,cluster_data in cluster_dict.items():
            cluster_list2.append(cluster_data[0])

        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print(cluster_list2)
        print("EEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")

        results_list = []

        for target_index in cluster_list2:
            print("<<<<<<<<<<<<<<< %s <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"%target_index)
            # results は target_index を含むクラスタを下から順に検査して、isomorphic thresholdをギリギリ超えない
            # ノードを探し、途中のノードを含むリストになっている
            results = self.findJustBeforeThresholdStepWiseFromLowerValue(target_index)
            # print(results)
            # resultsの最後に入っているものが狙っているもの
            # print(results)
            if len(results) > 0:
                results_list.append(results[-1])
            else:
                print("Not found")
        
        return results_list

if __name__ == "__main__":
    cctable = sys.argv[1]
    filename_list = sys.argv[2]
    cc2dendrogram = CC2Dendrogram(cctable, filename_list)
    # cc2dendrogram.drawDendrogram()
    # cc2dendrogram.rejectResort()
    # cc2dendrogram.divideClusterWithIsomorphicThreshold()
    cc2dendrogram.readData2()
