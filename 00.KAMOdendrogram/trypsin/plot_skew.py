import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# パラメータを設定
# Cluster 0072: alpha=-5.6823, loc=0.9868, scale=0.0187
# Cluster 0076: alpha=-11.0832, loc=0.9943, scale=0.0174
# Cluster 0072 and 0076: alpha=-9.9534, loc=0.9794, scale=0.0229
a1 = -5.6823
b1 = 0.9868
s1 = 0.0187
a2=-11.0832
b2=0.9943
s2=0.0174
a3=-9.9534
b3=0.9794
s3=0.0229

# X軸の範囲を設定
x = np.linspace(0.01, 1, 1000)

# 対数正規分布の確率密度関数を計算
y1 = skewnorm.pdf(x, a1, b1, s1)
y2 = skewnorm.pdf(x, a2, b2, s2)
y3 = skewnorm.pdf(x, a3, b3, s3)

# 対数正規分布をプロット
plt.plot(x, y1, label="Skewnorm AB")
plt.plot(x, y2, label="Skewnorm 0076")
plt.plot(x, y3, label="Skewnorm 0072")
plt.xlabel("x")
plt.xlim(0.9,1.0)
plt.ylabel("Density")
plt.legend()
plt.show()
