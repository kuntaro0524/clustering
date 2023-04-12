import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.stats import betaprime

#x = np.linspace(0, 100, 1)
x = np.arange(0,1,0.01)
print(x)

# beta関数を計算する
y_beta = beta.pdf(1-x, 1.5, 2.5,0,1)

# betaprime関数を計算する
y_betaprime = betaprime.pdf(1-x, 1.5,2.5)

# プロットする
print(y_beta)
plt.plot(x, y_beta, label="beta")
plt.plot(x, y_betaprime, label="betaprime")
#plt.legend()
plt.show()
