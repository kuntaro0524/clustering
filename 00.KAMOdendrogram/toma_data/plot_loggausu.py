import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# パラメータを設定
sigma1 = 0.5109
mean1 = -0.0014
scale1 = 0.0149

sigma2 = 1.1245
mean2 =  0.0048
scale2 = 0.0116

sigma3 = 0.6431
mean3 =  0.0001
scale3 = 0.0152

# X軸の範囲を設定
x = np.linspace(0.01, 1, 1000)

# 対数正規分布の確率密度関数を計算
y1 = lognorm.pdf(1-x, sigma1, mean1, scale=scale1)
y2 = lognorm.pdf(1-x, sigma2, mean2, scale=scale2)
y3 = lognorm.pdf(1-x, sigma3, mean3, scale=scale3)

# 対数正規分布をプロット
plt.plot(x, y1, label="Lognormal AB")
plt.plot(x, y2, label="Lognormal 0076")
plt.plot(x, y3, label="Lognormal 0072")
plt.xlabel("x")
plt.xlim(0.9,1.0)
plt.ylabel("Density")
plt.legend()
plt.show()
