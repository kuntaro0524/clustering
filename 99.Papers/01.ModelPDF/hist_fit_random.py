import numpy as np
from scipy.interpolate import Akima1DInterpolator
from multiprocessing import Pool, cpu_count
# pyplotをインポート
import matplotlib.pyplot as plt

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
# normをインポート
from scipy.stats import norm
pdf_true = norm.pdf(x_range, mu1, sigma1) + norm.pdf(x_range, mu2, sigma2)

# 補完された曲線を正規化し、確率密度関数として扱えるようにする。
x = np.linspace(bin_centers[0], bin_centers[-1], num=1000)
pdf = spline(x)
pdf = pdf / np.sum(pdf)  # スケーリングして確率密度関数にする

def random(x_range):
    x = np.random.uniform(bin_centers[0], bin_centers[-1])
    prob = spline(x)
    scaled_prob = prob / np.sum(spline(x_range))
    if np.random.rand() < scaled_prob:
        return x
    else:
        return random(x_range)

# 乱数生成
def random2(x):
    while True:
        x_ = np.random.uniform(x[0], x[-1])
        y_ = np.random.uniform(0, spline(x_))
        if y_ < spline(x_):
            return x_

def random4(x_range):
    x_ = np.linspace(x_range[0], x_range[1], 100)
    # 確率密度関数の累積分布関数 (CDF) を計算
    cdf = np.cumsum(pdf)
    cdf_ = cdf / cdf[-1]
    # make_interp_splineをインポート
    from scipy.interpolate import make_interp_spline
    spline = make_interp_spline(x_, cdf_)
    y_ = np.random.uniform(0, spline(x_range[1])) # 範囲を修正
    return y_

def random5(x_range):
    x_ = x_range[0]
    cdf_ = x_range[1]
    from scipy.interpolate import make_interp_spline
    spline = make_interp_spline(x_, cdf_)
    return spline(random(100))

def random3(x_range, n_samples=1):
    x_min, x_max = x_range
    x_ = np.linspace(x_min, x_max, 100)
    # interpolateをインポート
    from scipy import interpolate
    spline = interpolate.splrep(x, y, s=0)
    y_min = 0
    y_max = interpolate.splev(x_max, spline, der=0)
    y_ = np.random.uniform(y_min, y_max, n_samples)
    x_random = []
    y_random = []
    for y in y_:
        x_candidates = x_[spline[1] <= y]
        if len(x_candidates) > 0:
            x_random.append(np.random.choice(x_candidates))
            y_random.append(y)
    return np.array(x_random), np.array(y_random)

def random6(x_range):
    # 確率密度関数を生成
    x_ = np.linspace(x_range[0], x_range[1], 100)
    y_ = norm.pdf(x_, loc=2, scale=0.7)
    cdf_ = np.cumsum(y_)
    cdf_ /= cdf_[-1]
    
    # スプライン補間を行う
    from scipy.interpolate import make_interp_spline
    spline = make_interp_spline(x_, cdf_)
    samples = spline(np.random.random(1000))
    return samples

if __name__ == '__main__':
    pool = Pool(cpu_count())
    samples = pool.map(random6, [x_range]*1000)
    pool.close()
    pool.join()

    # random()関数を1000回呼び出して、ヒストグラムを生成
    plt.hist(samples, bins=bin_num, density=False)
    # 確率密度関数をプロット
    plt.plot(x, pdf * 10000, label='PDF')
    plt.plot(x_range, pdf_true, label='True PDF')
    plt.legend()
    plt.show()