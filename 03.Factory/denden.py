import numpy as np
import scipy.cluster.hierarchy as hierarchy

def get_cluster_labels(Z, index, data_labels):
    merge_info = Z[index]
    cluster1_index = int(merge_info[0])
    cluster2_index = int(merge_info[1])

    cluster1_data_indices = get_cluster_data_indices(Z, cluster1_index)
    cluster2_data_indices = get_cluster_data_indices(Z, cluster2_index)

    cluster1_labels = [data_labels[i] for i in cluster1_data_indices]
    cluster2_labels = [data_labels[i] for i in cluster2_data_indices]

    return cluster1_labels, cluster2_labels


def get_cluster_data_indices(Z, cluster_index):
    if cluster_index < len(Z):
        left_index = int(Z[cluster_index, 0])
        right_index = int(Z[cluster_index, 1])
        left_indices = get_cluster_data_indices(Z, left_index)
        right_indices = get_cluster_data_indices(Z, right_index)
        return left_indices + right_indices
    else:
        return [cluster_index - len(Z)]


# データラベルのリスト（例）
data_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# 例としてZとindexを指定
Z = hierarchy.linkage(dis_list, 'ward')
index = 10

# クラスタに含まれるデータのラベルを取得
cluster1_labels, cluster2_labels = get_cluster_labels(Z, index, data_labels)

print("Cluster 1 Labels:", cluster1_labels)
print("Cluster 2 Labels:", cluster2_labels)
