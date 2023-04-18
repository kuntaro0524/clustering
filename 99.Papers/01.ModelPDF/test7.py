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

from scipy.stats import rv_continuous
from scipy.integrate import quad

# 確率密度関数の定義
class MyPDF(rv_continuous):
    def _pdf(self, x):
        return spline(x)

# 確率分布オブジェクトの作成
my_dist = MyPDF(a=-np.inf, b=np.inf, name='my_distribution')

# 正規化
integral, _ = quad(my_pdf, -np.inf, np.inf)
norm_pdf = lambda x: my_pdf(x) / integral

# ランダム値の生成
random_values = my_dist.rvs(size=10)

pdf /= np.trapz(pdf, x)

# rv_continuousクラスを継承したクラスを作成し、確率密度関数を定義する。
class HistPDF(rv_continuous):
    def _pdf(self, x):
        return spl(x)

# 定義した確率密度関数に従って、ランダムに数値を取得する。
rv = HistPDF(a=bin_centers[0], b=bin_centers[-1], name='HistPDF')
samples = rv.rvs(size=1000)
print(samples)pdf /= np.trapz(pdf, x)

# rv_continuousクラスを継承したクラスを作成し、確率密度関数を定義する。
class HistPDF(rv_continuous):
    def _pdf(self, x):
        return spl(x)

# 定義した確率密度関数に従って、ランダムに数値を取得する。
rv = HistPDF(a=bin_centers[0], b=bin_centers[-1], name='HistPDF')
samples = rv.rvs(size=1000)
print(samples)