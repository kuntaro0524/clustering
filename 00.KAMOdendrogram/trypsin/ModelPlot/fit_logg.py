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
    
    parser = optparse.OptionParser()
    parser.add_option("-c", "--cluster1", dest="cluster1", default="0", help="cluster1")
    parser.add_option("-d", "--cluster2", dest="cluster2", default="1", help="cluster2")
    parser.add_option("-t", "--cc_threshold", dest="cc_threshold", default="0.8", help="cc_threshold")
    parser.add_option("-n", "--nbins", dest="nbins", default="20", help="nbins")
    parser.add_option("-p", "--process_type", dest="process_type", default="histogram", help="process_type")
    parser.add_option("-m", "--model_index", dest="model_index", default=0, help="model index 0: skewed, index 2: logg")
    
    (options, args) = parser.parse_args()

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

    # FittingVarious.model_funcs[1] を利用して数値列を作る
    # 0: skewed, 1: logg, 2: lognorm
    import numpy as np
    x = np.linspace(0, 1, 1000)
    ccModel.getModelFunctions()

    y_aa = ccModel.model_funcs[2](x, sigma_aa, loc_aa, scale_aa)

    # dfAAの 'cc' でヒストグラムを作成する
    import matplotlib.pyplot as plt
    plt.hist(dfAA['cc'], bins=int(len(dfAA['cc'])/int(options.nbins)), density=True, alpha=0.5, label='AA')
    plt.plot(x,y_aa)
    plt.xlim(0.9,1)
    plt.show()