import os,sys
import pandas as pd
import ProcCLUSTERSTXT as ptxt


def makecsv_of_target_cluster(filename, cluster_id):
    # filename: クラスタリング結果のファイル名
    # cluster_id: クラスタID
    # クラスタリング結果のファイルを読み込む
    p = ptxt.ProcCLUSTERINGTXT(filename)
    p.readClusters()
    # クラスタIDがcluster_idのデータを取得
    target_cluster = p.df[p.df["cluster_id"] == cluster_id]
    # クラスタIDがcluster_idのデータをCSVファイルに保存
    target_cluster.to_csv("cluster_" + str(cluster_id) + ".csv", index=False)

if __name__ == "__main__":
    filename = "CLUSTERS.txt"
    cluster_id = int(sys.argv[1])
    makecsv_of_target_cluster(filename, cluster_id)
    print("cluster_" + str(cluster_id) + ".csv is created.")
    print("Please check the file.")
    print("You can find the file in the current directory.")
    print(os.getcwd())
    print("Thank you.")
    sys.exit(0)