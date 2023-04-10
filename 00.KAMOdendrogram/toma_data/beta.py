import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import betaprime
from scipy.optimize import curve_fit

# CCデータの生成 (例としてランダムデータを生成)
np.random.seed(42)
cc_data = np.random.rand(1000)

# ヒストグラムを作成
num_bins = 50
hist, bin_edges = np.histogram(cc_data, bins=num_bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 第二種ベータ分布の確率密度関数
def betaprime_pdf(x, alpha, beta):
    return betaprime.pdf(x, alpha, beta)

# 第二種ベータ分布にフィット
initial_guess = (1, 1)  # 初期推定値（適切な値を選ぶことが重要です）
popt, pcov = curve_fit(betaprime_pdf, bin_centers, hist, p0=initial_guess)

# フィットした第二種ベータ分布をプロット
x = np.linspace(0, 1, 1000)
plt.plot(x, betaprime_pdf(x, *popt), 'r-', label="Fitted Beta Prime Distribution")

# CCデータのヒストグラムをプロット
plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.6, color='g', label="CC Data")
plt.xlabel("CC")
plt.ylabel("Density")
plt.legend()
plt.show()

