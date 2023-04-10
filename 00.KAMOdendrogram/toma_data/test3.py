import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# 歪ガウス関数の定義
def skewed_gaussian(x, mu, sigma, alpha):
    gauss = norm.pdf(x, mu, sigma)
    cdf = norm.cdf(alpha * (x - mu) / sigma)
    return 2 * gauss * cdf

# 仮の相関係数データを生成
np.random.seed(0)
data = np.random.normal(0.5, 0.2, 1000)

# ヒストグラムを作成し、binの中心と高さを取得
hist, bin_edges = np.histogram(data, bins=30, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 最適なパラメータを求める
initial_params = [0.5, 0.2, 0.0]  # mu, sigma, alpha
params, _ = curve_fit(skewed_gaussian, bin_centers, hist, p0=initial_params)
mu, sigma, alpha = params

# ヒストグラムをプロット
plt.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.6, color='g', label="Observed CC")

# 最適な歪ガウス関数をプロット
x = np.linspace(min(data), max(data), 1000)
y = skewed_gaussian(x, mu, sigma, alpha)
plt.plot(x, y, 'r', label="Fitted Skewed Gaussian")

# グラフの設定
plt.xlabel("CC")
plt.ylabel("Density")
plt.legend()
plt.show()

