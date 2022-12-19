import numpy as np
from matplotlib import pyplot as plt

cc_array=[]
cc_dist=[]
for cc in np.arange(0.0, 1.0, 0.02):
    cc_array.append(cc)
    cc_dist.append(np.sqrt(1-cc*cc))

plt.plot(cc_array,cc_dist,'o')
plt.savefig("cc_dist.jpg")
#plt.show()
