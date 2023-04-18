import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import rv_continuous

# 観測されたヒストグラムのデータを取得し、ヒストグラムを作成する。
# data はガウス関数に従ってランダムな誤差を含むものとする
np.random.seed(123)
data = np.concatenate([np.random.normal(loc=1, scale=1, size=1000),
                          np.random.normal(loc=2, scale=0.3, size=1000)])
hist, bin_edges = np.histogram(data, bins=5, density=False)

# スプライン補完を行い、補完された曲線を作成する。
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#spl = UnivariateSpline(bin_centers, hist, s=0, k=3)
# importする
from scipy.interpolate import Akima1DInterpolator
spl = Akima1DInterpolator(bin_centers, hist)

# 補完された曲線を正規化し、確率密度関数として扱えるようにする。
x = np.linspace(bin_centers[0], bin_centers[-1], num=1000)
pdf = spl(x)

# ヒストグラムと補完曲線を描く
import matplotlib.pyplot as plt
plt.hist(data, bins=100, density=True, alpha=0.5)
plt.plot(x, pdf)
plt.show()
 
pdf /= np.trapz(pdf, x)

# rv_continuousクラスを継承したクラスを作成し、確率密度関数を定義する。
class HistPDF(rv_continuous):
    def _pdf(self, x):
        return spl(x)

# 定義した確率密度関数に従って、ランダムに数値を取得する。
rv = HistPDF(a=bin_centers[0], b=bin_centers[-1], name='HistPDF')
samples = rv.rvs(size=10000)

import matplotlib.pyplot as plt
# １枚のグラフを水平方向に２枚にわける
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# もとのヒストグラムも描く
ax1.hist(data, bins=100, density=True, alpha=0.5)
# ヒストグラムを描く
ax2.hist(samples, bins=100, density=True, alpha=0.5)

plt.show()