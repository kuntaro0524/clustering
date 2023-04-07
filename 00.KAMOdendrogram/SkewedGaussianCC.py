#必要なモジュールをimportする
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import skewnorm
#minimize_scalarを使うためにscipy.optimizeをimportする
from scipy.optimize import minimize_scalar

#歪ガウス関数に関係するクラス
#利用するのはscipy.stats.skewnormである

class SkewedGaussianCC():
    def __init__(self, alpha, loc, scale):
        self.alpha = alpha
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return skewnorm.pdf(x, self.alpha, self.loc, self.scale)

    def cdf(self, x):
        return skewnorm.cdf(x, self.alpha, self.loc, self.scale)

    def ppf(self, x):
        return skewnorm.ppf(x, self.alpha, self.loc, self.scale)

    # 歪ガウス関数からランダムにサンプリングする
    # rvs: random variates
    def rvs(self, size=1):
        return skewnorm.rvs(self.alpha, self.loc, self.scale, size=size)

    # 歪ガウス関数のピーク座標を取得する
    def get_mode(self):
        # 最小化の対象となる関数（pdf関数を負にした関数）
        def neg_skewed_gaussian(x, alpha, loc, scale):
            return -skewnorm.pdf(x, alpha, loc, scale)

        # 最小化問題を解く
        result = minimize_scalar(neg_skewed_gaussian, args=(self.alpha, self.loc, self.scale))

        # ピーク座標（Mode）を取得
        mode_x = result.x
        mode_y = skewnorm.pdf(mode_x, self.alpha, self.loc, self.scale)
        #print(f"Mode position: x = {mode_x}, y = {mode_y}")
        return mode_x

    