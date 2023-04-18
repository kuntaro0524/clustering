from scipy.stats import rv_continuous
import numpy as np

class GaussianMixtureRV(rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussians = []
        
    def add_gaussian(self, amplitude, center, sigma):
        self.gaussians.append((amplitude, center, sigma))
        
    def _pdf(self, x):
        pdf = np.zeros_like(x)
        for amplitude, center, sigma in self.gaussians:
            pdf += amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)
        return pdf
    
gmm_rv = GaussianMixtureRV(name='gmm')
gmm_rv.add_gaussian(1.0, 0.0, 1.0)
gmm_rv.add_gaussian(0.5, 5.0, 0.5)

samples = gmm_rv.rvs(size=500)
print(samples)

# ヒストグラムを描く
import matplotlib.pyplot as plt
plt.hist(samples, bins=100, density=True, alpha=0.5)
plt.show()

pdf /= np.trapz(pdf, x)

# rv_continuousクラスを継承したクラスを作成し、確率密度関数を定義する。
class HistPDF(rv_continuous):
    def _pdf(self, x):
        return spl(x)

# 定義した確率密度関数に従って、ランダムに数値を取得する。
rv = HistPDF(a=bin_centers[0], b=bin_centers[-1], name='HistPDF')
samples = rv.rvs(size=1000)
print(samples)