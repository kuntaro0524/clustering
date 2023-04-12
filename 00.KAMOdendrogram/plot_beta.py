import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import curve_fit

# CCデータの生成 (例としてランダムデータを生成)
np.random.seed(42)
cc_data = np.random.rand(1000)

# ヒストグラムを作成
num_bins = 50
hist, bin_edges = np.histogram(cc_data, bins=num_bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 第二種ベータ分布の確率密度関数
def beta(x, alpha, beta):
    return betaprime.pdf(1-x, alpha, beta)

# 第二種ベータ分布にフィット
x=np.arange(0,1,0.001)

plt.plot(x, betaprime_pdf(x, 1.5,2.5), 'r-', label="Fitted Beta Prime Distribution")
plt.xlabel("CC")
plt.ylabel("Density")
plt.legend()
plt.show()

