import numpy as np
from scipy.stats import lognorm
# minimizeを使うためにimport
from scipy.optimize import minimize

# パラメータ
sigma = 0.931
loc = 0.0032
scale = 0.0163

import numpy as np
from scipy.stats import lognorm

# 確率密度関数を定義
def pdf(x):
    return lognorm.pdf(1-x, sigma, loc, scale)

# xをランダムに抽出する関数を定義
def sample():
    return 1 - lognorm.rvs(sigma, loc, scale)

# 確認用にxの値をいくつか抽出して表示
for i in range(10):
    print(sample())

# ヒストグラムの描画など、必要な処理を行う
x = np.linspace(0, 1, 1000)

ya=[]
for ex in x:
    ya.append(sample())

print(ya)

import matplotlib.pyplot as plt
plt.hist(ya)
plt.show()

plt.plot(x,ya)
plt.show()