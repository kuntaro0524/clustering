import os,sys
import pandas as pd

class ProcCLUSTERINGTXT:
    def __init__(self, filename):
        self.filename = filename
        self.clusters = []
        self.isRead = False

    def readClusters(self):
        data = []
        # Read the file
        with open(self.filename, "r") as file:
            lines = file.readlines()
            for line in lines[1:]:
                cluster = line.strip().split()
                # print(cluster)
                # data
                cluster_id = int(cluster[0])
                # n_datasets
                n_datasets = int(cluster[1])
                # Ward distance
                ward_distance = float(cluster[2])
                # data IDs (cluster[3:])
                data_ids = [int(x) for x in cluster[3:]]
                # print(len(data_ids))
                # dataframeを作成するためにdictのリストを作成
                data.append([cluster_id, n_datasets, ward_distance, data_ids])
        # Create a dataframe
        self.df = pd.DataFrame(data, columns=["cluster_id", "n_datasets", "ward_distance", "data_id"])
        self.isRead = True

    def estimateIsomorphicThreshold(self):
        if self.isRead == False:
            self.readClusters()
        # self.dfの中で、ward_distanceの最大値を求める
        max_ward_distance = self.df["ward_distance"].max()
        print(max_ward_distance)
        # max_ward_distance に 0.7 を賭けた数値を求め self.iso_threshold とする
        self.iso_threshold = max_ward_distance * 0.7
        print(self.iso_threshold)
        # ward ditance が self.iso_threshold 以下のcluster_idを求める
        isomorphic_clusters = self.df[self.df["ward_distance"] <= self.iso_threshold]
        print(isomorphic_clusters)
        # isomorphic ではないものも求める
        non_isomorphic_clusters = self.df[self.df["ward_distance"] > self.iso_threshold]
        print(non_isomorphic_clusters)
        # iromorphic_clusters と non_isomorphic_clusters の数を数える
        print("isomorphic_clusters: ", len(isomorphic_clusters))
        print("non_isomorphic_clusters: ", len(non_isomorphic_clusters))

    def containsOrNot(self, id1, id2):
        # id1 は cluster_id であり、id2も別のcluster_id
        # 今、データ数はid1 > id2 であるとする
        # id2の中の "data_ids"の要素がid1の中に含まれているかどうかを判定する
        # id1の中にid2の要素が含まれていればTrueを返す
        # そうでなければFalseを返す
        if self.isRead == False:
            self.readClusters()
        # id1のデータを取得
        data1 = self.df[self.df["cluster_id"] == id1]
        # id2のデータを取得
        data2 = self.df[self.df["cluster_id"] == id2]
        # id1の中にid2の要素が一つでも含まれていればTrueを返す
        data1_ids = data1["data_id"].values[0]
        data2_ids = data2["data_id"].values[0]
        for id in data2_ids:
            if id in data1_ids:
                return True

        return False

    def burasagariGroup(self, parent_id):
        if self.isRead == False:
            self.readClusters() 
        # parent_idと一致するDataFrameから data_idsを取得する
        parent_ids = self.df[self.df["cluster_id"] == parent_id]['data_id'].values[0]
        # parent id のWard distanceを取得する
        parent_ward_distance = self.df[self.df["cluster_id"] == parent_id]['ward_distance'].values[0]
        print("################")
        print(type(parent_ids))
        print("################")
        # self.dfに含まれる行を１行ずつ調査する
        # 各行のdata_idsの要素にparent_idsの要素が含まれているかどうかを調査する
        # 含まれている行のindexを取得する
        # そのindexを使って新しいDataFrameを作成する
        useful_indices = []
        for index, row in self.df.iterrows():
            data_ids = row['data_id']
            # ward distance が  parent_ward_distance よりも大きい場合は無視
            if row['ward_distance'] > parent_ward_distance:
                continue
                # もしも、parent_idsの要素がdata_idsの要素に含まれていれば useful_indicesにindexを追加する
            for id in parent_ids:
                if id in data_ids:
                    useful_indices.append(index)
                    break

        print(useful_indices)
        new_df = self.df.loc[useful_indices]
        # CSV fileに書き出す 
        # ファイル名には parent_id を含める
        filename = "cluster_" + str(parent_id) + ".csv"
        new_df.to_csv(filename, index=False)
        
        return new_df

    def getClusters(self):
        return self.clusters

    def printClusters(self):
        for cluster in self.clusters:
            print(cluster)

if __name__ == "__main__":
    filename = "CLUSTERS.txt"
    proc = ProcCLUSTERINGTXT(filename)
    #proc.readClusters()
    #print(proc.containsOrNot(11525, 11524))
    #print(proc.containsOrNot(11524, 10835))
    print(proc.burasagariGroup(int(sys.argv[1])))
    #proc.estimateIsomorphicThreshold()