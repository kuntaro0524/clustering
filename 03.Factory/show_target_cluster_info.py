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

for z in Z:
    print(z[0],z[1])

target_index= len(Z) - 1

def get_2indices(index):
    cluster_info = Z[index]
    starti=int(cluster_info[0])
    endi=int(cluster_info[1])
    return starti, endi

starti,endi = get_2indices(target_index)
print(starti,endi)

results_list = []
if starti < len(name_list):
    results_list.append(starti)
else:
    print("EEE")