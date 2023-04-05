import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.stats import skewnorm
from scipy.cluster.hierarchy import fcluster

# alpha, loc, scale
# AA
alpha_aa = -15.2177
loc_aa = 0.9945
scale_aa = 0.0195

# BB
alpha_bb = -8.4750
loc_bb = 0.9883
scale_bb = 0.0251

# AB
alpha_ab = -11.3798
loc_ab = 0.9799
scale_ab = 0.0277

from scipy.optimize import minimize_scalar

# scipy skewed gaussianをそれぞれ描く
# xの数値は0.0, 1.0とする
# AAの構造はalpha_aa, loc_aa, scale_aaをつかう
# BBの構造はalpha_bb, loc_bb, scale_bbをつかう
# ABの構造はalpha_ab, loc_ab, scale_abをつかう

x=np.linspace(0.9,1.0,10000)
print(x)

ab=skewnorm.pdf(x,alpha_ab,loc_ab,scale_ab)
aa=skewnorm.pdf(x,alpha_aa,loc_aa,scale_aa)
bb=skewnorm.pdf(x,alpha_bb,loc_bb,scale_bb)

# Fittingによるピーク座標の取得関数
def get_peak_code(alpha,loc,scale):
    # 最小化の対象となる関数（pdf関数を負にした関数）
    def neg_skewed_gaussian(x, alpha, loc, scale):
        return -skewnorm.pdf(x, alpha, loc, scale)

    # 最小化問題を解く
    result = minimize_scalar(neg_skewed_gaussian, args=(alpha, loc, scale))
    # X 範囲
    xdata = np.arange(0,1.1,0.001)
    ydata=-neg_skewed_gaussian(xdata,alpha,loc,scale)

    # ピーク座標（Mode）を取得
    mode_x = result.x
    mode_y = skewnorm.pdf(mode_x, alpha, loc, scale)
    print(f"Mode position: x = {mode_x}, y = {mode_y}")
    return mode_x

# AAのピーク座標を取得
aa_peak = get_peak_code(alpha_aa,loc_aa,scale_aa)
# ABのピーク座標を取得
ab_peak = get_peak_code(alpha_ab,loc_ab,scale_ab)
# BBのピーク座標を取得
bb_peak = get_peak_code(alpha_bb,loc_bb,scale_bb)

# AA - AB peakの差を計算
aa_ab_peak_diff = aa_peak - ab_peak
# BB - AB peakの差を計算
bb_ab_peak_diff = bb_peak - ab_peak

# プロットの中にそれぞれAA, AB, BBの３パラメータを表示する
plt.annotate(f"AA= {alpha_aa:.4f} {loc_aa:.4f} {scale_aa:.4f}" , xy=(0.2, 0.95), xycoords='axes fraction')
plt.annotate(f"BB= {alpha_bb:.4f} {loc_bb:.4f} {scale_bb:.4f}" , xy=(0.2, 0.90), xycoords='axes fraction')
plt.annotate(f"AB= {alpha_ab:.4f} {loc_ab:.4f} {scale_ab:.4f}" , xy=(0.2, 0.85), xycoords='axes fraction')

# プロットの中にそれぞれのピーク座標を表示する
plt.annotate(f"AA mode= {aa_peak:.4f}" , xy=(0.2, 0.75), xycoords='axes fraction')
plt.annotate(f"BB mode= {bb_peak:.4f}" , xy=(0.2, 0.70), xycoords='axes fraction')
plt.annotate(f"AB mode= {ab_peak:.4f}" , xy=(0.2, 0.65), xycoords='axes fraction')

# プロットの中にそれぞれAA-AB, BB-ABの差を表示する
plt.annotate(f"AA-AB= {aa_ab_peak_diff:.4f}" , xy=(0.2, 0.55), xycoords='axes fraction')
plt.annotate(f"BB-AB= {bb_ab_peak_diff:.4f}" , xy=(0.2, 0.50), xycoords='axes fraction')

# プロットの中に、x=aa_peak, x=ab_peak, x=bb_peakの線を引く
plt.axvline(x=aa_peak, color='blue', linestyle='dashed', alpha=0.5)
plt.axvline(x=bb_peak, color='green', linestyle='dashed', alpha=0.5)
plt.axvline(x=ab_peak, color='orange', linestyle='dashed', alpha=0.5)

# すべての数値をプロットする
plt.ylim(0,50)
plt.plot(x,aa,color="blue",label="AA")
plt.plot(x,bb,color='green',label="BB")
plt.plot(x,ab,color="orange",label="AB")
plt.legend()
plt.savefig("skewnorm.png")
plt.show()
