import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, fcluster

def add_labels_to_all_top_clusters(Z, ax, threshold):
    """
    指定したしきい値に基づいて形成される全てのトップレベルのクラスタに、
    Ward距離とクラスタ番号をラベルとして追加する関数。
    """
    # デンドログラムのデータを取得
    ddata = dendrogram(Z)

    # しきい値に基づいてトップレベルのクラスタを特定
    top_clusters = fcluster(Z, threshold, criterion='distance')
    unique_clusters = np.unique(top_clusters)

    # 各トップレベルクラスタの最も高いノードを見つける
    highest_nodes = {}
    for i, (x, y) in enumerate(zip(ddata['icoord'], ddata['dcoord'])):
        x_center = (x[1] + x[2]) / 2
        y_center = y[1]
        # このノードが属するクラスタのインデックスを取得
        cluster_idx = np.where(top_clusters == top_clusters[i // 10])[0][0]
        # 最も高いノードを更新
        if cluster_idx not in highest_nodes or y_center > highest_nodes[cluster_idx][1]:
            highest_nodes[cluster_idx] = (x_center, y_center, y_center)

    # 各トップレベルクラスタの最も高いノードにラベルを追加
    for idx, (x_center, _, y_center) in highest_nodes.items():
        cluster_label = idx + 1  # クラスタ番号
        ax.text(x_center, y_center, f"{y_center:.2f}\n({cluster_label})", ha='center', va='bottom', fontsize=8, color="blue")

    # しきい値に線を引く
    ax.axhline(y=threshold, c='red', linestyle='--')

# デンドログラムを描画
fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, ax=ax)

# しきい値を設定してラベルと線を追加
threshold = 5  # しきい値
add_labels_to_all_top_clusters(Z, ax, threshold)

plt.title('Hierarchical Clustering Dendrogram with Labels for All Top Clusters')
plt.xlabel('Sample index')
plt.ylabel('Distance (Ward)')
plt.show()
