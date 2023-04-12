import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, lognorm
# curve_fitを使うためにscipy.optimizeをimportする
from scipy.optimize import curve_fit

# データ生成
data = skewnorm.rvs(-15, 0.99, 0.02, size=1000)

# ヒストグラム作成
bins = np.exp(np.linspace(np.log(data.min()), np.log(data.max()), 20))
hist, edges = np.histogram(data, bins=bins)

# 中心の計算
bin_centers = (edges[:-1] + edges[1:]) / 2

# Lognormal分布のフィッティング
def lognorm_func(x, s, loc, scale):
    return lognorm.pdf(x, s, loc, scale)

p0 = (1, 0, 1)
popt, pcov = curve_fit(lognorm_func, bin_centers, hist, p0=p0)

# フィッティング結果のプロット
x = np.linspace(data.min(), data.max(), 1000)
plt.hist(data, bins=bins, density=True, alpha=0.5)
plt.plot(x, lognorm.pdf(x, *popt), 'r-', lw=2)
plt.show()
