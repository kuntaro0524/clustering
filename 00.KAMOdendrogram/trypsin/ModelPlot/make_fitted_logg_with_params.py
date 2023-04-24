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
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # 縦軸はヒストグラムのbin数でラベルを "Frequency" とする
    ax1.set_ylabel("Frequency", fontsize=18)
    # 軸のラベルなどフォントサイズを18ptsにする
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax3.tick_params(labelsize=14)
    
    # 横軸はCCの値でラベルを "CC" とする
    ax1.set_xlabel("CC", fontsize=14)
    ax2.set_xlabel("CC", fontsize=14)
    ax3.set_xlabel("CC", fontsize=14)
    # タイトルを "AA" とする
    # ylabelのみ少し左へ移動
    #ax1.set_title("AA", fontsize=18, y=-0.2)
    ax1.set_ylabel("Frequency", fontsize=14, labelpad=14)
    ax2.set_ylabel("Frequency", fontsize=14, labelpad=14)
    ax3.set_ylabel("Frequency", fontsize=14, labelpad=14)

    # ラベルサイズは18ptsとする
    ax1.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    ax3.tick_params(labelsize=14)
    # fontsizerを18ptsにする
    # title について CC(apo-apo) なのだが、（）内を下付きにする
    # そのためには、"$CC_{apo-apo}$" とする
    ax1.set_title("$CC_{apo-apo}$", fontsize=18)
    ax2.set_title("$CC_{benz-benz}$", fontsize=18)
    ax3.set_title("$CC_{apo-benz}$", fontsize=18) 

    # グラフどうしの間隔を調整する
    fig.subplots_adjust(wspace=0.5)
    # グラフの下側の隙間を調整する
    fig.subplots_adjust(bottom=0.2)
    
    # FittingVarious.model_funcs[1] を利用して数値列を作る
    # 0: skewed, 1: logg, 2: lognorm

    import numpy as np
    x = np.linspace(0, 1, 1000)
    ccModel.getModelFunctions()
    y_aa = ccModel.model_funcs[2](x, sigma_aa, loc_aa, scale_aa)
    # dfAAの 'cc' でヒストグラムを作成する
    ax1.hist(dfAA['cc'], bins=110, density=False, alpha=0.5, label='AA')
    ax1.plot(x,y_aa,alpha=0.7)
    ax1.set_xlim(0.8,1)

    # FittingVarious.model_funcs[1] を利用して数値列を作る
    # 0: skewed, 1: logg, 2: lognorm
    x = np.linspace(0, 1, 1000)
    y_bb = ccModel.model_funcs[2](x, sigma_bb, loc_bb, scale_bb)
    # dfAAの 'cc' でヒストグラムを作成する
    ax2.hist(dfBB['cc'], bins=80, density=False, alpha=0.5, label='BB')
    ax2.plot(x,y_bb,alpha=0.7)
    ax2.set_xlim(0.8,1)

    # FittingVarious.model_funcs[1] を利用して数値列を作る
    # 0: skewed, 1: logg, 2: lognorm
    x = np.linspace(0, 1, 1000)
    y_ab = ccModel.model_funcs[2](x, sigma_ab, loc_ab, scale_ab)
    # dfABの 'cc' でヒストグラムを作成する
    ax3.hist(dfAB['cc'], bins=200, density=False, alpha=0.5, label='AB')
    ax3.plot(x,y_ab,alpha=0.7)
    ax3.set_xlim(0.8,1)
    plt.savefig("S21a.png")
    plt.show()