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

# 補完された曲線を正規化し、確率密度関数として扱えるようにする。
x = np.linspace(bin_centers[0], bin_centers[-1], num=1000)
pdf = spline(x)
pdf = pdf / np.sum(pdf)  # スケーリングして確率密度関数にする

# 確率密度関数に従ってランダムな数値を生成する関数を定義
def random():
    x_range = np.linspace(-8, 10, 1000)
    while True:
        print("cycle")
        # 範囲から一つの数値を生成
        x = np.random.uniform(bin_centers[0], bin_centers[-1])
        # ランダムに生成された数値が確率密度関数のどの位置にあるかを計算
        prob = spline(x)
        # スケーリングした確率密度関数の値を計算
        scaled_prob = prob / np.sum(spline(x_range))  # 確率密度関数の面積が1になるようにスケーリング
        # 乱数を生成して確率密度関数の位置と比較する
        if np.random.rand() < scaled_prob:
            return x

# random()関数を1000回呼び出して、ヒストグラムを生成
plt.hist([random() for _ in range(1000)], bins=bin_num, density=False)
# 確率密度関数をプロット
plt.plot(x, pdf * 10000, label='PDF')
plt.show()