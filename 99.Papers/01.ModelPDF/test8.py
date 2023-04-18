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
data = np.concatenate([x1, x2])
hist, bins = np.histogram(data, bins=bin_num, density=True)
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

# 補完された曲線を正規化し、確率密度関数として扱えるようにする。
x = np.linspace(bin_centers[0], bin_centers[-1], num=1000)
pdf = spline(x)

# ヒストグラムと補完曲線を描く
import matplotlib.pyplot as plt
plt.hist(data, bins=100, density=False, alpha=0.5)
plt.plot(x, pdf)
 
pdf /= np.trapz(pdf, x)

# rv_continuousクラスを継承したクラスを作成し、確率密度関数を定義する。
# rv_continuousクラスは、scipy.statsモジュールに含まれる。
from scipy.stats import rv_continuous
class HistPDF(rv_continuous):
    def _pdf(self, x):
        return spline(x)

# 定義した確率密度関数に従って、ランダムに数値を取得する。
rv = HistPDF(a=bin_centers[0], b=bin_centers[-1], name='HistPDF')
samples = rv.rvs(size=1000)
print(samples)

import matplotlib.pyplot as plt
# １枚のグラフを水平方向に２枚にわける
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# もとのヒストグラムも描く
ax1.hist(data, bins=100, density=True, alpha=0.5)
# ヒストグラムを描く
ax2.hist(samples, bins=100, density=True, alpha=0.5)

plt.show()
