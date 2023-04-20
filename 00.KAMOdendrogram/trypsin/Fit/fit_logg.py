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

    if options.process_type == "histogram":
        ccModel.plotHistOnly(options.cluster1, options.cluster2, float(options.cc_threshold), int(options.nbins))
    elif options.process_type == "logscale":
        ccModel.runLogscale(options.cluster1, options.cluster2, float(options.cc_threshold), int(options.nbins))
    elif options.process_type == "fit_various":
        ccModel.fitAll(options.cluster1, options.cluster2, float(options.cc_threshold), int(options.nbins))
    elif options.process_type == "test":
        # thresholdの数値を表示する
        ccModel.testRun(options.cluster1, options.cluster2, float(options.cc_threshold), int(options.nbins))
    elif options.process_type == "ffff":
        # thresholdの数値を表示する
        ccModel.fitOnly(options.cluster1, options.cluster2, float(options.cc_threshold), int(options.nbins), int(options.model_index))
