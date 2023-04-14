import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import lognorm

# パラメータの初期値
sigma, loc, scale = 0.931, 0.0032, 0.0163

# 指定した数だけxをランダム抽出
num_samples = 1000
rvs = lognorm.rvs(sigma, loc, scale, size=num_samples)

# 抽出したxを1-xに反転
x = 1 - rvs

# xの分布を確認するためにヒストグラムをプロット
bins = np.linspace(0, 1, 50)
plt.xlim(0.9,1.0)
plt.hist(x, bins=200)
plt.show()

plt.plot(x,y)
