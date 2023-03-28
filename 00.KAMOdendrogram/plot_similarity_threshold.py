import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

data = [
    (100, 0.60020, 0.01039),
    (200, 0.65260, 0.01700),
    (500, 0.73299, 0.01123),
    (1000, 0.79268, 0.01159),
]

n_total = [row[0] for row in data]
threshold = [row[1] for row in data]
sigma = [row[2] for row in data]

# CubicSplineを使って補間します
cs = CubicSpline(n_total, threshold)

# 補間するための新しいx軸の値を生成します
n_total_new = np.linspace(n_total[0], n_total[-1], 1000)

# 新しいx軸の値に対応する補間されたy軸の値を計算します
threshold_new = cs(n_total_new)

# エラーバー付きの散布図をプロットします
plt.errorbar(n_total, threshold, yerr=sigma, fmt="o-", capsize=5, elinewidth=2, markeredgewidth=2, label="Data")


plt.xlim(0,1050)
#plt.axhline(0.7,color='r',linestyle="--")

all_range=np.arange(0,10000)
plt.fill_between(all_range, 0.6, 0.7, color="pink", alpha=0.3, label="0.6 < y < 0.7")


# 軸ラベルを追加します
plt.xlabel("# of datasets")
plt.ylabel("similarity threshold")

# 凡例を追加します
plt.legend()

# グラフを表示します
plt.savefig("sim_thresh.jpg")
