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

# dis_listがnumpy配列であると仮定
# 無限大の要素をチェック
inf_indices = np.where(np.isinf(dis_list))
print("無限大の要素の位置：", inf_indices)

# NaNの要素をチェック
nan_indices = np.where(np.isnan(dis_list))
print("NaNの要素の位置：", nan_indices)

# dis_listがnumpy配列であると仮定
# 無限大やNaNを含むかどうかをチェック
if not np.all(np.isfinite(dis_list)):
    print("replace inf and nan to 0")
    # 無限大やNaNを別の値で置き換える（例えば0や平均値など）
    dis_list = np.nan_to_num(dis_list)


# その後、hierarchy.linkageを実行
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
dn = hierarchy.dendrogram(Z,labels=name_list)
plt.savefig("dendrogram.jpg")
plt.show()