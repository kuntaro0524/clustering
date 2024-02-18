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
            for parent_id in parent_ids:
                # print("!!!",parent_id)
                if parent_id in data_ids:
                    # print(data_ids)
                    useful_indices.append(index)
                    break

        print(useful_indices)
        new_df = self.df.loc[useful_indices]
        # CSV fileに書き出す 
        new_df.to_csv("new_df.csv", index=False)
        
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
    print(proc.burasagariGroup(11523))