import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

file_name = "cc.dat"

with open(file_name)as f:
    lines = f.readlines()

cc_list=[]
dis_list=[]

for line in lines[0:]:
    cc_list.append(float(line))
    dis_list.append(np.sqrt(1-float(line)**2))

Z = hierarchy.linkage(dis_list, 'ward')

# name list
file_name2 = sys.argv[1]
with open(file_name2) as f2:
    lines = f2.readlines()

name_list=[]
for line in lines:
    name_list.append(line.strip())

fig=plt.figure(figsize=(36,24))
ax=plt.gca()

ax.tick_params(axis='x', which='major', labelsize=30)
ax.tick_params(axis='y', which='major', labelsize=30)

dn = hierarchy.dendrogram(Z,ax=ax,labels=name_list,leaf_font_size=18)

fig.savefig("dendrogram.jpg")
#plt.show()
