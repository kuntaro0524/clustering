import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# データを生成する
np.random.seed(123)
data = np.concatenate([np.random.normal(loc=1, scale=1, size=1000),
                       np.random.normal(loc=2, scale=0.3, size=1000)])

# ヒストグラムを描画する
plt.hist(data, bins=100, density=True, alpha=0.5)
plt.show()

# ヒストグラムの情報を取得する
counts, bins = np.histogram(data, bins=100, density=True)

# ビンの中心を計算する
bin_centers = (bins[:-1] + bins[1:]) / 2

# binsに対してcountsをプロットする
print("######################")
plt.plot(bin_centers, counts, label='histogram')
print("######################")

# スプライン補間を行う
# spl = UnivariateSpline(bin_centers, counts, s=0, k=5)
# 一次元のスプライン補間を行う
spl = UnivariateSpline(bin_centers, counts, s=1, k=3)

# スプライン補間をプロットする
x_plot = np.linspace(-5, 5, num=1000)
plt.plot(x_plot, spl(x_plot), label='spline')

# 確率密度関数を定義する
my_func = spl.antiderivative()

# 0から10までのx座標で確率密度関数をプロットする
#x_plot = np.linspace(0, 100, num=1000)
x_plot = np.linspace(-5, 5, num=1000)
plt.xlim(-6,6)
plt.ylim(0,1)

plt.plot(x_plot, my_func(x_plot), label='cdf')
# 凡例を表示する
plt.legend()

plt.show()