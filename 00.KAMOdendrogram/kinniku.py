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
    pack_ID = line.split('/')[-4]
    arange = line.split('/')[-2]
    name_list.append(str(pack_ID) + '/' +str(arange))
Z = hierarchy.linkage(dis_list, 'ward')
plt.figure()
dn = hierarchy.dendrogram(Z,labels=name_list)
plt.show()
plt.savefig("sample.png")
