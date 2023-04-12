import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm, skewnorm, lognorm

# フィッティングに使用する関数
def func_1(x, a, b, c):
    return a * norm.pdf(x, b, c)

def func_2(x, a, b, c):
    return a * skewnorm.pdf(x, b, c)

def func_3(x, a, b, c):
    return a * lognorm.pdf(x, b, c)

# モデル関数のリスト
model_funcs = [func_1, func_2, func_3]

# CCデータの生成 (例としてランダムデータを生成)
np.random.seed(42)
cc_data = np.random.rand(1000)

# ヒストグラムを作成
num_bins = 40
hist, bin_edges = np.histogram(cc_data, bins=num_bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 各モデル関数に対する初期推定値
initial_guesses = [(1, 0.5, 0.1), (1, 0, 1), (1, 0, 0.5)]

# 各モデル関数をフィットして、AICを計算
aic_list = []
popt_list = []
for func, initial_guess in zip(model_funcs, initial_guesses):
    #popt, pcov = curve_fit(func, bin_centers, hist, p0=initial_guess)
    popt, pcov = curve_fit(func, bin_centers, hist, p0=initial_guess, method='lm', maxfev=100000)

    popt_list.append(popt)
    residuals = hist - func(bin_centers, *popt)
    sse = np.sum(residuals**2)
    aic = 2 * len(initial_guess) + len(bin_centers) * np.log(sse / len(bin_centers))
    aic_list.append(aic)

# AICが最小のモデル関数を選択
selected_index = np.argmin(aic_list)
selected_func = model_funcs[selected_index]
selected_popt = popt_list[selected_index]

# フィットしたモデル関数をプロット
x = np.linspace(0, 1, 1000)
plt.plot(x, selected_func(x, *selected_popt), 'r-', label="Selected Model")

# CCデータのヒストグラムをプロット
plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.6, color='g', label="CC Data")
plt.xlabel("CC")
plt.ylabel("Density")
plt.legend()
plt.show()

