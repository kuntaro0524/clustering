import sys
import matplotlib.pyplot as plt
import pandas as pd

# 複数のCSVファイルを読む
# python p.py *.csv
# とすると、複数のCSVファイルを読み込む
# ただし、1つのCSVファイルのみの場合は、
# python p.py *.csv
# とすると、エラーになる
# そのため、以下のようにする
    
filelist = sys.argv[1:]
if len(filelist) != 1:

    for i, f in enumerate(filelist):
        print("processing %s"% f)
        # CSVファイルを読み込む
        # 含まれるデータは以下
        # nds,delta_loc,scale_scale,threshold,new_threshold,filename,cluster_1_A_purity,cluster_1_B_purity,cluster_1_count,cluster_2_count,score
        df = pd.read_csv(f)
        # delta_locごとにグループ化してscoreおよびnew_thresholdの平均値と標準偏差を出す
        # また、new_thresholdは平均値のみ
        df = df.groupby("delta_loc").agg({"score": ["mean", "std"], "new_threshold": "mean"})

        # dfをカラム"new_threshold"の値の小さい順にソート
        #df = df.sort_values("new_threshold")
        
        # グラフを描画
        # x軸は new_threshold y軸はscore
        # エラーバーは標準偏差
        plt.errorbar(df["new_threshold"], df["score"]["mean"], yerr=df["score"]["std"], label=f)
        plt.plot(df["new_threshold"], df["score"]["mean"],'o')
        
        # legend labelは　filename にする
        # new_threshold に対して score をプロット
        plt.ylabel("Score")
        plt.xlabel("New Threshold")
        plt.legend()
        #plt.savefig("score_%d.png"%i)
        #plt.clf()
    plt.show()
else:
    f = filelist[0]
    print("processing %s"% f)
    # CSVファイルを読み込む
    # 含まれるデータは以下
    # nds,delta_loc,scale_scale,threshold,new_threshold,filename,cluster_1_A_purity,cluster_1_B_purity,cluster_1_count,cluster_2_count,score
    df = pd.read_csv(f)
    # delta_locごとにグループ化してscoreおよびnew_thresholdの平均値と標準偏差を出す
    # また、new_thresholdは平均値のみ
    df = df.groupby("delta_loc").agg({"score": ["mean", "std"], "new_threshold": "mean"})

    # dfをカラム"new_threshold"の値の小さい順にソート
    #df = df.sort_values("new_threshold")
    
    # グラフを描画
    # x軸は new_threshold y軸はscore
    # エラーバーは標準偏差
    plt.errorbar(df["new_threshold"], df["score"]["mean"], yerr=df["score"]["std"], label=f)
    plt.plot(df["new_threshold"], df["score"]["mean"],'o')
    
    # legend labelは　filename にする
    # new_threshold に対して score をプロット
    plt.ylabel("Score")
    plt.xlabel("New Threshold")
    plt.legend()
    plt.savefig("score.png")
    plt.clf()
    
plt.show()

