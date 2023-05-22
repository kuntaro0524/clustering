import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

file_name = sys.argv[1]

with open(file_name)as f:
    lines = f.readlines()
cc_list=[]
dis_list=[]

for line in lines[1:]:
    x = line.split()
    cc_list.append(float(x[2]))
    dis_list.append(np.sqrt(1-float(x[2])**2))

##label
file_name2 = sys.argv[2]
with open(file_name2) as f2:
    lines = f2.readlines()

name_list = []

for line in lines:
    name_list.append(line.strip())


Z = hierarchy.linkage(dis_list, 'ward')

# 最後から１つ目で、一番高い山のWard distanceを取得
last_merge = Z[-1]
thresh0 = last_merge[2]

# 最後のから２つ目、で、一番高い山のWard distanceを取得
last_merge = Z[-2]  
threshold = last_merge[2]  

# 新しい threshold
max_thresh = threshold
print("max_thresh",thresh0)
new_thresh = threshold/thresh0

print(new_thresh)

plt.figure()
# 各ノードの番号を含むラベルリストを作成
#cluster_labels = [str(i) for i in range(1, len(dis_list) + 1)]
dn = hierarchy.dendrogram(Z,labels=name_list)

# クラスタ番号を付けるためのリストを作成
cluster_labels = [str(i) for i in range(1, len(dn['ivl']) + 1)]
ax = plt.gca()
ax.set_xticklabels(cluster_labels)

# fclusterを利用して各ノードの解析をしたいのだが
from scipy.cluster.hierarchy import fcluster

# クラスタのしきい値を設定
t = 2.5

# 各データ点のクラスタを取得
labels = fcluster(Z, t, criterion='distance')

cluster_data = {}
for i, label in enumerate(labels):
    if label not in cluster_data:
        cluster_data[label] = []
    cluster_data[label].append(i)

for label, data_indexes in cluster_data.items():
    print('label', label)
    print('data_indexes', data_indexes)
    print(len(data_indexes))
    # print('data', [name_list[i] for i in data_indexes])
    print('')

import matplotlib.pyplot as plt

# デンドログラムの生成
dendrogram = hierarchy.dendrogram(Z, link_color_func=lambda k: 'k')

threshold = 2.5

# 分岐点のWard距離の数値を表示するための関数
def add_distance_labels(ax, linkage_matrix, threshold):
    for i, d, c in zip(linkage_matrix[:, 0], linkage_matrix[:, 2], linkage_matrix[:, 3]):
        x = 0.5 * (dendrogram['icoord'][i][1] + dendrogram['icoord'][i][2])
        y = d
        if y > threshold:
            ax.text(x, y, f'{y:.2f}', color='r')
        else:
            ax.text(x, y, f'{y:.2f}', color='k')

# 分岐点のWard距離の数値を表示
fig, ax = plt.subplots()
add_distance_labels(ax, Z, threshold=5)  # 適切なしきい値を設定
plt.show()
