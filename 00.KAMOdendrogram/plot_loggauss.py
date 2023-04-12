import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import betaprime
from scipy.stats import lognorm

#x = np.linspace(0, 100, 1)
sigma1 = 0.5109
mean1 = -0.0014
scale1 = 0.0149

x = np.arange(0,1,0.01)
print(x)

# beta関数を計算する
y_norm = lognorm.pdf(1-x, sigma1,mean1,scale1)

plt.plot(x, y_norm, label="beta")
plt.show()
