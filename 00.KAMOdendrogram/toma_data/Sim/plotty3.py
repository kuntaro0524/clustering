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
#plot_name = "cluster_1_count"
#plot_name = "threshold"
if len(filelist) != 1:
    plt.figure(figsize=(10,10))

    # color indexを作っておく
    color_index = 0
    # color list
    color_list = ["blue", "green", "black", "purple", "orange", "pink", "brown", "gray"]
    for i, f in enumerate(filelist):
        # file名:f からデータ数を取り出す
        # nds_1000.csv ならば 1000
        nds = int(f.split("_")[1].split(".")[0])
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
        # fontsize は 20
        plt.tick_params(labelsize=20)
        plt.plot(df[plot_name], df["score"]["mean"]/0.6,'o-',label=f'{nds} sets',color=color_list[color_index])
        # それぞれ最小のthresholdのプロットだけサイズの大きいX点をプロット 
        plt.plot(df[plot_name].iloc[0], df["score"]["mean"].iloc[0]/0.6,'X',markersize=12,color=color_list[color_index])
        plt.ylabel("Score", fontsize=20)
        plt.xlabel("Isomorphic threshold", fontsize=20)
        
        #plt.plot(df['delta_loc'], df["score"]["mean"],'o-',label=f)
        plt.legend(fontsize=20)
        color_index += 1

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
    # fontはすべて24
    plt.ylabel("Score", fontsize=48)
    plt.xlabel("Isomorphic threshold", fontsize=48)
    plt.legend(fontsize=48)
    plt.savefig("score.png")
    plt.clf()

plt.savefig("score.png")
plt.show()