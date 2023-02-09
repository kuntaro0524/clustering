import numpy as np
import sys, os

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

for idx1 in np.arange(0,ndata):
    for idx2 in np.arange(idx1+1,ndata):
        cc_value=float(lines[all_idx].strip())
        print(cc_value)
        # Identical 1
        if idx1<50 and idx2<50:
            aa_list.append(cc_value)
        # Different
        elif (idx1<50 and idx2>=50) or (idx1>=50 and idx2<50):
            ab_list.append(cc_value)
        # Identical 2
        elif (idx1>=50 and idx2>=50):
            bb_list.append(cc_value)
        all_idx+=1


aaa=np.array(aa_list)
aba=np.array(ab_list)
bba=np.array(bb_list)

print(aaa.mean(),aaa.std(),np.median(aaa))
print(aba.mean(),aba.std(),np.median(aba))
print(bba.mean(),bba.std(),np.median(bba))

from matplotlib import pyplot as plt
plt.hist(aaa,bins=20,alpha=0.5)
plt.hist(aba,bins=20,alpha=0.5)
plt.hist(bba,bins=20,alpha=0.5)
plt.show()
