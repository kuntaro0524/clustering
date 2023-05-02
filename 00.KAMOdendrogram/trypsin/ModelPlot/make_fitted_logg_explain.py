# -*- coding: utf-8 -*-
# ２つ上のディレクトリにあるFittingVarious.pyをimportする
import sys
sys.path.append('../../')
import FittingVarious

if __name__ == "__main__":

    # instance
    ccModel = FittingVarious.FittingVarious()

    cluster_numbers = sys.argv[1:]
    cc_values_list = []

    # optparseをimport
    import optparse
    # コマンドラインから以下の引数を取得する
    # option1: クラスタ番号1
    # option2: クラスタ番号2
    # option3: CCの閾値
    # option4: ヒストグラムのbin数の割合
    # optparse で引数を取得する
    # option5: 処理のタイプ
    # "histgram" : ヒストグラムのみ表示
    # "logscale" : ヒストグラムとCCの分布を表示
    

    dfAA,dfBB,dfAB=ccModel.extractCCs("cctable.dat", "filenames.lst",0.8)

    # Trypsinのやつは最適値を選択している
    # nbins=85
    aaa=[0.7044,0.0027,0.0131]
    bba=[0.7648,0.0060,0.0247]
    aba=[0.7871,0.0147,0.0250]
    
    sigma_aa = aaa[0]
    loc_aa = aaa[1]
    scale_aa = aaa[2]
    
    sigma_bb = bba[0]
    loc_bb = bba[1]
    scale_bb = bba[2]
    
    sigma_ab = aba[0]
    loc_ab = aba[1]
    scale_ab = aba[2]

    # これからプロットを作成するが
    # 水平方向にプロットは３つ並べる
    # AAのヒストグラムとモデル関数のプロットを重ねる
    # BBのヒストグラムとモデル関数のプロットを重ねる
    # ABのヒストグラムとモデル関数のプロットを重ねる
    # それぞれのプロットの横軸はCCの値
    # 縦軸はヒストグラムのbin数
    # また、モデル関数のプロットは、
    # ヒストグラムのbin数の割合に応じて、
    # 1000個の点を作成してプロットする
    # 例えば、bin数が100の場合は、
    # 0.0, 0.01, 0.02, ... , 0.99, 1.0 の1000個の点を作成する
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'  # フォントの種類
    fig = plt.figure(figsize=(20,20))

    # 横軸はCCの値でラベルを "CC" とする
    plt.xlabel("CC", fontsize=14)
    plt.ylabel("Frequency", fontsize=14, labelpad=14)

    # ラベルサイズは18ptsとする
    # ticsの数値は消す
    plt.axis("off")

    import numpy as np
    x = np.linspace(0, 1, 1000)
    ccModel.getModelFunctions()
    y_aa = ccModel.model_funcs[2](x, sigma_aa, loc_aa, scale_aa)
    # dfAAの 'cc' でヒストグラムを作成する
    label_AB = "$CC_{A-A}$"
    plt.plot(x,y_aa,alpha=0.6, label=label_AB, linewidth=15, color="blue")

    # FittingVarious.model_funcs[1] を利用して数値列を作る
    # 0: skewed, 1: logg, 2: lognorm
    x = np.linspace(0, 1, 1000)
    y_bb = ccModel.model_funcs[2](x, sigma_bb, loc_bb, scale_bb)
    # dfAAの 'cc' でヒストグラムを作成する
    label_BB = "$CC_{B-B}$"
    plt.plot(x,y_bb,alpha=0.6, label=label_BB, linewidth=15, color="orange")

    # FittingVarious.model_funcs[1] を利用して数値列を作る
    # 0: skewed, 1: logg, 2: lognorm
    x = np.linspace(0, 1, 10000)
    y_ab = ccModel.model_funcs[2](x, sigma_ab, loc_ab, scale_ab)
    # dfABの 'cc' でヒストグラムを作成する
    label_AB = "$CC_{A-B}$"
    plt.plot(x,y_ab,alpha=0.6, label=label_AB, linewidth=15, color="green")
    # legendを表示する
    #plt.legend(loc="upper left", fontsize=48)
    plt.xlim(0.9,1)
    #plt.title("CC models(Probability Density Function)", fontsize=48)
    plt.savefig("explain.png")
    plt.show()

    # 積分値を求める
    from scipy.integrate import quad
    # Set 1
    # lognorm
    from scipy.stats import lognorm
    p1, _ = quad(lambda x: lognorm.pdf(1 - x, sigma_aa, loc_aa, scale_aa), 0, np.inf)
    print("AA 1 probability:", p1)
    
    # Set 2
    p2, _ = quad(lambda x: lognorm.pdf(1 - x, sigma_bb, loc_bb, scale_bb), 0, np.inf)
    print("BB 2 probability:", p2)
    
    # Set 3
    p3, _ = quad(lambda x: lognorm.pdf(1 - x, sigma_ab, loc_ab, scale_ab), 0, np.inf)
    print("AB 3 probability:", p3)