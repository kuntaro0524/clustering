import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, fcluster


def add_labels_to_each_cluster_corrected(Z, ax, threshold):
    """
    指定したしきい値に基づいて形成される全てのクラスタに対して、
    最も高いノードにWard距離とクラスタ番号をラベルとして追加する関数。
    """
    # デンドログラムのデータを取得
    ddata = dendrogram(Z)

    # しきい値に基づいてクラスタを特定
    clusters = fcluster(Z, threshold, criterion='distance')
    cluster_ids = np.unique(clusters)

    # 各クラスタの最も高いノード（クラスタ形成の場所）を特定
    cluster_heights = {cid: 0 for cid in cluster_ids}
    for i, (x, y) in enumerate(zip(ddata['icoord'], ddata['dcoord'])):
        x_center = (x[1] + x[2]) / 2
        y_center = y[1]
        leaf_idx = int(i / 2)
        cluster_id = clusters[leaf_idx]
        if y_center > cluster_heights[cluster_id]:
            cluster_heights[cluster_id] = y_center

    # 各クラスタの最も高いノードにラベルを追加
    for i, (x, y) in enumerate(zip(ddata['icoord'], ddata['dcoord'])):
        x_center = (x[1] + x[2]) / 2
        y_center = y[1]
        leaf_idx = int(i / 2)
        cluster_id = clusters[leaf_idx]
        if y_center == cluster_heights[cluster_id]:
            ax.text(x_center, y_center, f"{y_center:.2f}\n({cluster_id})", ha='center', va='bottom', fontsize=8, color="blue")

    # しきい値に線を引く
    ax.axhline(y=threshold, c='red', linestyle='--')

# デンドログラムを描画
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, ax=ax)

# しきい値を設定してラベルと線を追加
threshold = 5  # しきい値
add_labels_to_each_cluster_corrected(Z, ax, threshold)

plt.title('Hierarchical Clustering Dendrogram with Labels for Each Cluster at Threshold')
plt.xlabel('Sample index')
plt.ylabel('Distance (Ward)')
plt.show()
