import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import curve_fit

# ベータ分布の関数を定義
def beta_pdf(x, a, b, loc, scale):
    return beta.pdf(x, a, b, loc, scale)

# 仮の相関係数データを生成（ベータ分布に従う）
np.random.seed(0)
data = beta.rvs(2, 5, size=1000)

# ヒストグラムを作成し、binの中心と高さを取得
hist, bin_edges = np.histogram(data, bins=30, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 最適なパラメータを求める
initial_params = [1, 1, 0, 1]  # a, b, loc, scale
params, _ = curve_fit(beta_pdf, bin_centers, hist, p0=initial_params)
a, b, loc, scale = params

# ヒストグラムをプロット
plt.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.6, color='g', label="Observed CC")

# 最適なベータ分布をプロット
x = np.linspace(min(data), max(data), 1000)
y = beta_pdf(x, a, b, loc, scale)
plt.plot(x, y, 'r', label="Fitted Beta Distribution")

# グラフの設定
plt.xlabel("CC")
plt.ylabel("Density")
plt.legend()
plt.show()

