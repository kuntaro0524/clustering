import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

file_name = sys.argv[1]
df = pd.read_csv(file_name)

for line in lines[1:]:
    x = line.split()
    cc_list.append(float(x[2]))

print(cc_list)

# Histgram of CC
fig = plt.figure(figsize=(25,10))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) #この1行を入れる

plt.hist(cc_list,bins=50)
plt.show()
