import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

# set
cl_number = 3
name1 = 'apo'
name2 = 'ben'

# distance matrix
file_name = sys.argv[1]

with open(file_name)as f:
    lines = f.readlines()

dis_list = []

for line in lines[1:]:
    x = line.split()
    dis_list.append(np.sqrt(1-float(x[2])**2))


# label
file_name2 = sys.argv[2]

with open(file_name2) as f2:
    lines = f2.readlines()

name_list = []

for line in lines:
    pack_ID = line.split('/')[-4]
    arange = line.split('/')[-2]
    name_list.append(str(pack_ID) + '/' + str(arange))

print(len(name_list))


# change label
label2 = []
for name in lines:
    print(name)
    if name.rfind("apo") != -1:
        label2.append(name1)
    else:
        label2.append(name2)

# print(len(label2))

Z = hierarchy.linkage(dis_list, 'ward')

# search
result = []
nod = len(name_list)  # number of data
cl = len(name_list) + len(Z) - cl_number
temp = [cl]
result = []
while len(temp) != 0:
    for t in temp:
        if nod <= t:
            cl_list = [Z[int(t)-nod, 0], Z[int(t)-nod, 1]]
            for c in cl_list:
                if nod <= c:
                    temp.append(c)
                else:
                    result.append(c)
            temp.remove(t)
        else:
            result.append(t)
            temp.remove(t)


# count
count_list = []
for i in result:
    count_list.append(label2[int(i)])

noa = count_list.count(name1)
nob = count_list.count(name2)

print(f"{name1} : {noa}")
print(f"{name2} : {nob}")

dn = hierarchy.dendrogram(Z, labels=label2, leaf_font_size=10)
plt.show()
