import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# パラメータを設定
sigma = 1.5 # 対数正規分布の形状パラメータ
mean = np.log(0.9)   # 対数正規分布の平均
scale = np.exp(mean)  # 対数正規分布のスケールパラメータ

# X軸の範囲を設定
x = np.linspace(0.01, 1, 1000)

# 対数正規分布の確率密度関数を計算
y = lognorm.pdf(1-x, sigma, loc=0, scale=scale)

# 対数正規分布をプロット
plt.plot(x, y, label="Lognormal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()

