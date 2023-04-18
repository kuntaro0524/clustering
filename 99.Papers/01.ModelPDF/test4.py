import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import Akima1DInterpolator

# 2つの正規分布を生成
mu1, sigma1 = -2, 1
mu2, sigma2 = 3, 2
x1 = np.random.normal(mu1, sigma1, 500)
x2 = np.random.normal(mu2, sigma2, 500)

# ヒストグラムを生成
bin_num = 50
hist, bins = np.histogram(np.concatenate([x1, x2]), bins=bin_num, density=True)
bin_centers = (bins[1:] + bins[:-1]) / 2

# Akima補間曲線を生成
spline = Akima1DInterpolator(bin_centers, hist)

# 確率密度関数をプロット
x_range = np.linspace(-8, 10, 1000)
plt.plot(x_range, spline(x_range), label='PDF')
plt.hist(np.concatenate([x1, x2]), bins=bin_num, density=True, alpha=0.5, label='Histogram')
plt.plot(x_range, norm.pdf(x_range, mu1, sigma1) + norm.pdf(x_range, mu2, sigma2), label='True PDF')
plt.legend()
plt.show()

# 補完された曲線を正規化し、確率密度関数として扱えるようにする。
x = np.linspace(bin_centers[0], bin_centers[-1], num=1000)
pdf = spline(x)
