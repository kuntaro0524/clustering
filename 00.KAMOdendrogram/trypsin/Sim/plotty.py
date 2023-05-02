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

plot_name = "new_threshold"
#plot_name = "threshold"
if len(filelist) != 1:
    plt.figure(figsize=(10,10))

    for i, f in enumerate(filelist):
        print("processing %s"% f)
        # CSVファイルを読み込む
        # 含まれるデータは以下
        # nds,delta_loc,scale_scale,threshold,new_threshold,filename,cluster_1_A_purity,cluster_1_B_purity,cluster_1_count,cluster_2_count,score
        df = pd.read_csv(f)
        # delta_locごとにグループ化してscore, threshold, new_threshold, cluster_1_A_purity, cluster_1_B_purity, cluster_1_count, cluster_2_countの平均値と標準偏差を出す
        # また、new_threshold, thresholdは平均値のみ
        df = df.groupby("delta_loc").agg({"score": ["mean", "std"], "threshold": "mean", "new_threshold": "mean", "cluster_1_A_purity": ["mean", "std"], "cluster_1_B_purity": ["mean", "std"], "cluster_1_count": ["mean", "std"], "cluster_2_count": ["mean", "std"]})
        print(df)

        # グラフを描画
        # グラフサイズは 10x10
        # new_threshold に対して score をプロット
        #plt.errorbar(df[plot_name], df["score"]["mean"], yerr=df["score"]["std"], label=f)
        # 'plot_name' の小さい順にソート
        df = df.sort_values(by=[(plot_name,'mean')])
        #plt.plot(df[plot_name], df["score"]["mean"],'o-',label=f)
        # delta_locをx軸にする
        #plt.plot(df.index, df["score"]["mean"],'o-',label=f)
        plt.plot(df[plot_name], df["score"]["mean"],'o-',label=f)
        plt.ylabel("Score")
        plt.xlabel(plot_name)
        
        #plt.plot(df['delta_loc'], df["score"]["mean"],'o-',label=f)
        plt.legend()

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

plt.savefig("score.png")
plt.show()