import numpy as np

irange = np.arange(0,100,1)

for i in irange:
    if i == 0:
        continue
    ndata = i*(i-1) /2.0
    print(i,ndata)