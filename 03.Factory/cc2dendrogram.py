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
#plt.savefig("dendrogram.jpg")
plt.show()

leaf_indexes = dn['leaves']
for index in leaf_indexes:
    print(name_list[index])

print(Z)
for index,z in enumerate(Z):
    print(index,z)