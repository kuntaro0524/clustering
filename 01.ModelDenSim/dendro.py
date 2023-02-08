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
plt.figure()
dn = hierarchy.dendrogram(Z)
plt.savefig("dendrogram.jpg")
plt.show()
