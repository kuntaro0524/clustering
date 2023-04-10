import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.optimize import curve_fit

# CCデータの生成 (例としてランダムデータを生成)
np.random.seed(42)
cc_data = np.random.rand(1000)

# ビン数とビンの幅を設定
num_bins = 10
bin_width = 1 / num_bins

# CCデータを離散化（ビン分け）
hist, bin_edges = np.histogram(cc_data, bins=num_bins, range=(0, 1))
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 二項分布の確率質量関数
def binom_pmf(x, n, p):
    return binom.pmf(x, n, p)

# 二項分布にフィット
popt, pcov = curve_fit(binom_pmf, np.arange(num_bins), hist)

# フィットした二項分布をプロット
#x = np.arange(num_bins)
x = np.arange(0,1,0.001)

plt.plot(bin_centers, binom_pmf(x, *popt), 'r-', label="Fitted Binomial Distribution")

# CCデータのヒストグラムをプロット
plt.bar(bin_centers, hist, width=bin_width, alpha=0.6, color='g', label="CC Data")
plt.xlabel("CC")
plt.ylabel("Frequency")
plt.legend()
plt.show()
