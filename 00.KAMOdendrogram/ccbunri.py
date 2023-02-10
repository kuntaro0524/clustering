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

# Structure1/Structure2
st1="apo"
st2="ben"

for line in lines:
    pathname = line.strip()
    if pathname.rfind(st1)!=-1:
        name_list.append(st1)
    else:
        name_list.append(st2)

print(name_list)

index=0
apoapo=[]
benben=[]
apoben=[]

for (i,first) in enumerate(name_list):
    for (j,second) in enumerate(name_list[i+1:]):
        if first=="apo" and second=="apo":
            apoapo.append(cc_list[index])
        elif first=="ben" and second=="ben":
            benben.append(cc_list[index])
        else:
            apoben.append(cc_list[index])
        index+=1

# Histgram of CC
fig = plt.figure(figsize=(25,10))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) #この1行を入れる

plt.hist([apoapo,benben,apoben],bins=50)
plt.show()
