import numpy as np
from matplotlib  import pyplot as plt

lines = open("data.txt","r").readlines()

value_list=[]
for line in lines:
    value = float(line.strip())
    value_list.append(value)

plt.hist(value_list)
plt.show()
