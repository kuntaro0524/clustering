import sys
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

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

# 距離行列から階層構造を計算
Z = sch.linkage(dis_list, 'ward')

# Zに含まれているのはなにか？
# print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
# print(Z.shape)
#１行ずつZを表示する
# for i in range(Z.shape[0]):
    # print(Z[i])
# print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")

# デンドログラムの生成
fig, ax = plt.subplots()
# figure の大きさを大きくする
fig = plt.figure(figsize=(100, 80))
# figure の両端の余白をなくす
fig.subplots_adjust(left=0.0, right=1.0)
ax.tick_params(axis='x', which='major', labelsize=35)

# name_listを文字列処理する
# name_listの中から puckid, pinid を抜き出す
# 文字列は
# /user/target/Auto/2023A/BL32XU_230424_Asada/_kamoproc/merge_ccc_methods2_2.53S_S1PR3_TY/input_files/CPS1899-04/data00/CPS1899-04-multi_801-900/XDS_ASCII.HKL_noscale
# まず、/で分割する
# 最後から２番めにある文字列を抜き出す
# これをname_list2に格納する
name_list2 = []
for i in range(len(name_list)):
    tmp_name = name_list[i].split('/')[-2]
    # "multi"の文字列で分割する
    tmp_name = tmp_name.split('multi')[0]
    name_list2.append(tmp_name)

    # すでにname_list2にtmp_nameがあるかどうかをチェックする
    # "_"が入っているかチェックし、ある場合には "_" で分割して後側の数字を読む(tmp_num)
    # 前側はname_list2に入っているかチェックする
    # このとき数字で読めなければエラーで止める
    # tmp_numに１加えて、
        
# fontsize を大きくする
# dendrogram = sch.dendrogram(Z,labels=name_list)
# labelサイズを大きくする
dendrogram = sch.dendrogram(Z, labels=name_list2, leaf_font_size=10)
# dendrogram = sch.dendrogram(Z, leaf_font_size=10)

# 分岐点のWard距離の数値を表示する関数
def add_distance_labels(ax, linkage_matrix, labels, wd_min, wd_max):
    for i, (d, label) in enumerate(zip(linkage_matrix[:, 2], labels)):
        x = dendrogram['icoord'][i][1]
        y = dendrogram['dcoord'][i][1]
        if wd_min <= d <= wd_max:
            ax.text(x, y, f'{d:.2f}', color='r', ha='center', va='bottom')
            ax.text(x, y, label, color='k', ha='center', va='top')

# 分岐点のWard距離の数値とインデックス番号を表示
wd_min = 2.0  # Ward距離の最小値
wd_max = 5.0  # Ward距離の最大値
add_distance_labels(ax, Z, dendrogram['ivl'], wd_min, wd_max)
plt.savefig("dendrogram.png")
plt.show()

# クラスタごとにデータ名のグループを表示
def print_cluster_groups(linkage_matrix, labels):
    clusters = sch.fcluster(linkage_matrix, t=wd_max, criterion='distance')
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"length of clusters: {len(clusters)}")
    for cl in clusters:
        print(cl)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    unique_clusters = np.unique(clusters)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"length of unique_clusters: {len(unique_clusters)}")
    for cl in unique_clusters:
        print(cl)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    for cluster in unique_clusters:
        group_indices = np.where(clusters == cluster)[0]
        group_labels = [labels[i] for i in group_indices]
        print(f'Cluster {cluster}: {group_labels}')

print_cluster_groups(Z, dendrogram['ivl'])