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

# 各ノードの番号を含むラベルリストを作成
#cluster_labels = [str(i) for i in range(1, len(dis_list) + 1)]
# ax = plt.gca()
plt.figure()
dn = hierarchy.dendrogram(Z,labels=name_list)

# クラスタ番号を付けるためのリストを作成
cluster_labels = [str(i) for i in range(1, len(dn['ivl']) + 1)]
#ax = plt.gca()
#ax.set_xticklabels(cluster_labels)

# fclusterを利用して各ノードの解析をしたいのだが
from scipy.cluster.hierarchy import fcluster

# クラスタのしきい値を設定
t = 2.0

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

# 分岐点のX座標とY座標を取得
x_coordinates = [0.5 * (x[1] + x[2]) for x in dn['icoord']]
#y_coordinates = dn['dcoord'][:, 1]
y_coordinates = [y[1] for y in dn['dcoord']]

print("TEST",len(x_coordinates))
print("TEST2",len(cluster_data))
print("MAX_THRESH=",thresh0)

for idx,(x,y) in enumerate(zip(x_coordinates, y_coordinates)):
    # isomorphic threshold 
    it_tmp = y/thresh0
    # y (高さ)が閾値を超えている場合は、その点をプロット
    if y > t:
        plt.text(x, y, 'index:{:5d},IT={:.2f},WD={:.2f}'.format(idx,it_tmp,y), ha='center', va='bottom')

plt.show()