import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import betaprime

# パラメータを設定
a1 = 3.5744
b1 = 244.1206
a2 = 3.0001
b2 = 179
a3 = 2.5
b3 = 165

# X軸の範囲を設定
x = np.linspace(0.01, 1, 1000)

# 対数正規分布の確率密度関数を計算
y1 = betaprime.pdf(1-x, a1, b1)
y2 = betaprime.pdf(1-x, a2, b2)
y3 = betaprime.pdf(1-x, a3, b3)

# 対数正規分布をプロット
plt.plot(x, y1, label="Lognormal AB")
plt.plot(x, y2, label="Lognormal 0076")
plt.plot(x, y3, label="Lognormal 0072")
plt.xlabel("x")
plt.xlim(0.9,1.0)
plt.ylabel("Density")
plt.legend()
plt.show()
