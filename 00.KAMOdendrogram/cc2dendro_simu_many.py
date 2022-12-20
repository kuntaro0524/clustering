import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

# CC(apo-apo)   Average: 0.925913, variance: 0.009192, median: 0.968700
# CC(benz-benz) Average: 0.917110, variance: 0.008674, median: 0.957200
# CC(apo-benz)  Average: 0.907780, variance: 0.008920, median: 0.947350

sample_dict=[{"name":"apo-apo","mean":0.958,"sigma":0.009192},
             {"name":"apo-benz", "mean":0.9472,"sigma":0.00867},
             {"name":"benz-benz","mean":0.957,"sigma":0.00892}]

sample_dict=[{"name":"apo-apo","mean":0.958,"sigma":0.05},
             {"name":"apo-benz", "mean":0.9472,"sigma":0.05},
             {"name":"benz-benz","mean":0.957,"sigma":0.05}]


def get_stat_info(cc_combination):
    for idx,s in enumerate(sample_dict):
        if s['name']==cc_combination:
            return s

def make_random_cc(stat_dict):
    mean=stat_dict['mean']
    sigma=stat_dict['sigma']
    while(True):
        randcc = np.random.normal(loc=mean,scale=sigma)
        if randcc >=0 and randcc<=1.0:
            break

    return randcc

sample_list=[]
for i in np.arange(0,50):
    sample_list.append("apo")
for i in np.arange(0,50):
    sample_list.append("benz")

dis_list = []
name_list=[]
for idx1,s1 in enumerate(sample_list):
    for s2 in sample_list[idx1+1:]:
        if s1=="apo" and s2=="apo":
            stat_dict=get_stat_info("apo-apo")
            name_list.append("apo-apo")
        elif s1=="benz" and s2=="benz":
            stat_dict=get_stat_info("benz-benz")
            name_list.append("benz-benz")
        else:
            stat_dict=get_stat_info("apo-benz")
            name_list.append("apo-benz")
        cctmp = make_random_cc(stat_dict)
        dist = np.sqrt(1-cctmp*cctmp)
        dis_list.append(dist)

print(dis_list,len(dis_list))
Z = hierarchy.linkage(dis_list, 'ward')
plt.figure()
# dn = hierarchy.dendrogram(Z,labels=name_list)
dn = hierarchy.dendrogram(Z)
#plt.savefig("dendrogram.jpg")
plt.show()
