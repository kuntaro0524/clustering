import numpy as np
import sys, os
from scipy.cluster import hierarchy


lines = open("cc.dat","r").readlines()
ndata = int(sys.argv[1])

# N data in a file
n_in_file = len(lines)

expected_nfiles=int(ndata*(ndata-1)/2)

if n_in_file != expected_nfiles:
    print("Something wrong")

all_idx=0
data_idx=0

rows=int(ndata/2.0)
print(rows)
columns=3

# 0: AA, 1: AB, 2:BB
aa_list=[]
ab_list=[]
bb_list=[]

dist_list=[]
name_list=[]
for idx1 in np.arange(0,ndata):
    for idx2 in np.arange(idx1+1,ndata):
        cc_value=float(lines[all_idx].strip())
        # Identical 1
        if idx1<50 and idx2<50:
            aa_list.append(cc_value)
        # Different
        elif (idx1<50 and idx2>=50) or (idx1>=50 and idx2<50):
            ab_list.append(cc_value)
        # Identical 2
        elif (idx1>=50 and idx2>=50):
            bb_list.append(cc_value)
    
        # Distance list from CC table
        dist_tmp = np.sqrt(1-cc_value*cc_value)
        dist_list.append(dist_tmp)
        all_idx+=1

aaa=np.array(aa_list)
aba=np.array(ab_list)
bba=np.array(bb_list)

print(aaa.mean(),aaa.std(),np.median(aaa))
print(aba.mean(),aba.std(),np.median(aba))
print(bba.mean(),bba.std(),np.median(bba))

outfile=open("results.dat","w")
outfile.write("AA(mean,std,median)=%12.5f %12.5f %12.5f\n"% (aaa.mean(), aaa.std(), np.median(aaa)))
outfile.write("AB(mean,std,median)=%12.5f %12.5f %12.5f\n"% (aba.mean(), aba.std(), np.median(aba)))
outfile.write("BB(mean,std,median)=%12.5f %12.5f %12.5f\n"% (bba.mean(), bba.std(), np.median(bba)))
outfile.close()

from matplotlib import pyplot as plt
plt.hist(aaa,bins=20,alpha=0.5)
plt.hist(aba,bins=20,alpha=0.5)
plt.hist(bba,bins=20,alpha=0.5)
plt.savefig("cc_dist.png")

Z = hierarchy.linkage(dist_list, 'ward')

# name list
file_name2 = "name.lst"
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

fig.savefig("dendro.jpg")
