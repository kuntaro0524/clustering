import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
import sys
sys.setrecursionlimit(10**6)  # 任意の大きな値に変更可能

# 観測されたヒストグラムを生成
mu1, sigma1 = -2, 1
mu2, sigma2 = 3, 2
x1 = np.random.normal(mu1, sigma1, 500)
x2 = np.random.normal(mu2, sigma2, 500)
hist, bins = np.histogram(np.concatenate([x1, x2]), bins=50, density=True)
bin_centers = (bins[1:] + bins[:-1]) / 2

# スプライン補間を行い、確率密度関数として正規化する
spline = Akima1DInterpolator(bin_centers, hist)
x_range = np.linspace(bin_centers[0], bin_centers[-1], num=1000)
pdf = spline(x_range)
pdf = pdf / np.sum(pdf)

# 確率密度関数をプロット
#plt.plot(x_range, pdf)
#plt.show()


from scipy.stats import rv_continuous

# MyDistributionクラスを定義し、確率密度関数を定義する
class MyDistribution(rv_continuous):
    def _pdf(self, x):
        return spline(x) / np.sum(pdf)

# MyDistributionクラスのインスタンスを生成し、ランダムな数値を取り出す関数を定義する
mydist = MyDistribution(a=bin_centers[0], b=bin_centers[-1])
def random():
    return mydist.rvs()


# 1000回のサンプリングを行い、ヒストグラムをプロットする
alist = []
for i in range(0,1000):
    print(i)
    tmpd = random()
    alist.append(tmpd)

data = np.array(alist)

plt.hist(data, bins=50, density=True, alpha=0.5, label='Histogram')
plt.show()
#plt.show()    

from multiprocessing import Pool

def random():
    return mydist.rvs()

if __name__ == '__main__':
    with Pool(processes=8) as pool:
        results = pool.map(random, range(1000))

data = np.array(results)

# 結果をプロット
plt.hist(data, bins=50, density=True, alpha=0.5, label='Histogram')
plt.show()