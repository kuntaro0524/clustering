# csvファイルを読み込んでpandas dataframeに格納する
# 含まれるカラムは
import pandas as pd
import matplotlib.pyplot as plt
import sys


fig = plt.figure(figsize=(15,15))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) #この1行を入れる

def evaluateScore(filename, pattern=1):
    if pattern==1:
        # nds, delta_loc, scale_scale, threshold, new_threshold, filename, cluster_1_A_purity, cluster_1_B_purity, cluster_1_count, cluster_2_count, score
        df = pd.read_csv(filename)
        # 各行でスコアを以下のように計算する
        #  min(N1_A/NA, N1_B/NB, N2_A/NA, N2_B/NB) * (2 - abs(N1_A/NA - 0.5) - abs(N1_B/NB - 0.5) - abs(N2_A/NA - 0.5) - abs(N2_B/NB - 0.5))
        # N1_Aは cluster_1_A_purity * cluster_1_count
        # N1_Bは cluster_1_B_purity * cluster_1_count
        # N2_Aは (1 - cluster_1_A_purity) * cluster_2_count
        # N2_Bは (1 - cluster_1_B_purity) * cluster_2_count
        # NAは nds/2
        # NBは nds/2
        # このスコアを全ての行について計算し、平均を出力する
        df["N1_A"] = df["cluster_1_A_purity"] * df["cluster_1_count"]
        df["N1_B"] = df["cluster_1_B_purity"] * df["cluster_1_count"]
        df["N2_A"] = (1 - df["cluster_1_A_purity"]) * df["cluster_2_count"]
        df["N2_B"] = (1 - df["cluster_1_B_purity"]) * df["cluster_2_count"]
        df["NA"] = df["nds"] / 2
        df["NB"] = df["nds"] / 2
        df["score"] = df[["N1_A", "N1_B", "N2_A", "N2_B"]].min(axis=1) * (2 - abs(df["N1_A"] / df["NA"] - 0.5) - abs(df["N1_B"] / df["NB"] - 0.5) - abs(df["N2_A"] / df["NA"] - 0.5) - abs(df["N2_B"] / df["NB"] - 0.5))
        print(df['score'])
        return df

    elif pattern==2:
        df = pd.read_csv(filename)
        # 各行でスコアを以下のように計算する
        # 1-abs(n1-n2)/(n1+n2)
        # n1は cluster_1_count
        # n2は cluster_2_count
        # このスコアを全ての行について計算し、平均を出力する
        df["score"] = 1 - abs(df["cluster_1_count"] - df["cluster_2_count"]) / (df["cluster_1_count"] + df["cluster_2_count"])
        print(df['score'])
        return df

    elif pattern==3:
        # New score
        df = pd.read_csv(filename)
        # 新しいスコアの定義は以下の通り
        # barance_score = 1 - abs(n1-n2)/(n1+n2)
        df["balance_score"] = 1 - abs(df["cluster_1_count"] - df["cluster_2_count"]) / (df["cluster_1_count"] + df["cluster_2_count"])
        # purity_score については、cluster_1_A_purityとcluster_1_B_purityのうち大きい方をとる
        df["purity_score"] = df[["cluster_1_A_purity", "cluster_1_B_purity"]].max(axis=1)
        # alphaを重みとしたときtotal_scoreは以下のようになる
        # total_score = alpha * balance_score + (1 - alpha) * purity_score
        alpha = 0.9
        df["total_score"] = alpha * df["balance_score"] + (1-alpha) * df["purity_score"]

        # delta_loc が同じものをまとめる
        # まとめたものの平均をとる
        #new_df = df.groupby('delta_loc').mean()
        # それぞれラベルはデータセット数でcsv_file名から抜き取る
        # csv_file名は例えば、"./nds_100.csv" という形式である
        # これを100 という数値として取得する
        # これをラベルとしてプロットする
        label_name = filename.split("_")[1].split(".")[0]
        # それぞれのスコアをプロットする
        plt.plot(df['new_threshold'], df["balance_score"], color=color_list[color_index], label="balance_score_" + label_name)
        plt.show()

        # filenameのうち .csv拡張子を除いたものをプレフィクスとする
        prefix = filename.replace(".csv","")
        # 新たなCSVファイルを作成
        df.to_csv(f"{prefix}_score.csv")

        return df

    elif pattern==10:
        # New score
        df = pd.read_csv(filename)
        # 成功の是非について以下のように定義する
        # 結果は１もしくは０である
        # １は成功、０は失敗
        # balance_score = 1 - abs(n1-n2)/(n1+n2)
        # 分母が0の場合がある失敗とする
        # cluster_1_countとcluster2_countの和が0の場合は失敗とする
        # この数値が0.8以上であれば成功、そうでなければ失敗
        # purity_score については
        # cluster_1_A_purityとcluster_1_B_purityを比較して大きい方を cluster1_scoreとする
        # cluster_2_A_purityとcluster_2_B_purityを比較して大きい方を cluster2_scoreとする
        # この２つのうち大きい方を purity_scoreとする
        # この数値が0.8以上であれば成功、そうでなければ失敗
        # dataframeに含まれる行を１行ずつ処理する
        for index, row in df.iterrows():
            # cluster_1_countとcluster2_countの和が0の場合は失敗とする
            if row["cluster_1_count"] + row["cluster_2_count"] == 0:
                df.loc[index, "balance_score"] = 0
            else:
                df.loc[index, "balance_score"] = 1 - abs(row["cluster_1_count"] - row["cluster_2_count"]) / (row["cluster_1_count"] + row["cluster_2_count"])
            # cluster_1_A_purityとcluster_1_B_purityを比較して大きい方を cluster1_scoreとする
            if row["cluster_1_A_purity"] > row["cluster_1_B_purity"]:
                cluster1_score = row["cluster_1_A_purity"]
            else:
                cluster1_score = row["cluster_1_B_purity"]
            # cluster_2_A_purityとcluster_2_B_purityを比較して大きい方を cluster2_scoreとする
            if row["cluster_2_A_purity"] > row["cluster_2_B_purity"]:
                cluster2_score = row["cluster_2_A_purity"]
            else:
                cluster2_score = row["cluster_2_B_purity"]
            # この２つのうち大きい方を purity_scoreとする
            if cluster1_score > cluster2_score:
                df.loc[index, "purity_score"] = cluster1_score
            else:
                df.loc[index, "purity_score"] = cluster2_score
            # purity_score が 0.8未満の場合は失敗
            # balance_score が 0.8未満の場合は失敗
            if df.loc[index, "purity_score"] < 0.8 or df.loc[index, "balance_score"] < 0.8:
                df.loc[index, "result"] = 0
            else:
                df.loc[index, "result"] = 1

        return df

    elif pattern==11:
        # New score
        df = pd.read_csv(filename)
        # 成功の是非について以下のように定義する
        # 結果は１もしくは０である
        # １は成功、０は失敗
        # balance_score = 1 - abs(n1-n2)/(n1+n2)
        # 分母が0の場合がある失敗とする
        # cluster_1_countとcluster2_countの和が0の場合は失敗とする
        # この数値が0.8以上であれば成功、そうでなければ失敗
        # purity_score については
        # cluster_1_A_purityとcluster_1_B_purityを比較して大きい方を cluster1_scoreとする
        # cluster_2_A_purityとcluster_2_B_purityを比較して大きい方を cluster2_scoreとする
        # この２つのうち大きい方を purity_scoreとする
        # この数値が0.8以上であれば成功、そうでなければ失敗
        # dataframeに含まれる行を１行ずつ処理する
        for index, row in df.iterrows():
            # cluster_1_countとcluster2_countの和が0の場合は失敗とする
            if row["cluster_1_count"] + row["cluster_2_count"] == 0:
                df.loc[index, "balance_score"] = 0
            else:
                df.loc[index, "balance_score"] = 1 - abs(row["cluster_1_count"] - row["cluster_2_count"]) / (row["cluster_1_count"] + row["cluster_2_count"])
            # cluster_1_A_purityとcluster_1_B_purityを比較して大きい方を cluster1_scoreとする
            if row["cluster_1_A_purity"] > row["cluster_1_B_purity"]:
                cluster1_score = row["cluster_1_A_purity"]
            else:
                cluster1_score = row["cluster_1_B_purity"]
            # cluster_2_A_purityとcluster_2_B_purityを比較して大きい方を cluster2_scoreとする
            if row["cluster_2_A_purity"] > row["cluster_2_B_purity"]:
                cluster2_score = row["cluster_2_A_purity"]
            else:
                cluster2_score = row["cluster_2_B_purity"]
            # この２つのうち小さい方を
            if cluster1_score < cluster2_score:
                df.loc[index, "purity_score"] = cluster1_score
            else:
                df.loc[index, "purity_score"] = cluster2_score
            # total score
            df.loc[index, "total_score"] = 0.5*df.loc[index, "balance_score"] + 0.5*df.loc[index, "purity_score"]

        print(df)
        return df

# 複数のCSVを読み込む : argvから読む
csv_files = sys.argv[1:]

# color index を作る
color_index = 0
color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
pattern = 11
for csv_file in csv_files:
    df = evaluateScore(csv_file, pattern=pattern)

    # それぞれラベルはデータセット数でcsv_file名から抜き取る
    # csv_file名は例えば、"./nds_100.csv" という形式である
    # これを100 という数値として取得する
    # これをラベルとしてプロットする
    label_name = csv_file.split("_")[1].split(".")[0]
    
    # delta_loc ごとに scoreをプロットする
    # new_threshold に 対してscoreをプロットする
    if pattern==3:
        # score が 0 のものは除外する
        df = df[df['total_score'] != 0]
        # delta_loc が同じものをまとめる
        new_df = df.groupby('delta_loc').mean()
        plt.plot(new_df['new_threshold'], new_df['total_score'],'o-',color=color_list[color_index],label=label_name)
        # それぞれのプロットの最初の点は "X" で表示する
        plt.plot(new_df['new_threshold'].iloc[0], new_df['total_score'].iloc[0], 'x', color=color_list[color_index], markersize=15)
        # X labelは "isomorphic threshold" 18pts
        plt.xlabel("isomorphic threshold", fontsize=18)
        # Y labelは "score" 18
        plt.ylabel("balance score", fontsize=18)

    if pattern==2:
        # score が 0 のものは除外する
        df = df[df['score'] != 0]
        # delta_loc が同じものをまとめる
        new_df = df.groupby('delta_loc').mean()
        plt.plot(new_df['new_threshold'], new_df['score'],'o-',color=color_list[color_index],label=label_name)
        # それぞれのプロットの最初の点は "X" で表示する
        plt.plot(new_df['new_threshold'].iloc[0], new_df['score'].iloc[0], 'x', color=color_list[color_index], markersize=15)
        # X labelは "isomorphic threshold" 18pts
        plt.xlabel("isomorphic threshold", fontsize=18)
        # Y labelは "score" 18
        plt.ylabel("balance score", fontsize=18)

    if pattern==9:
        plot_name = "new_threshold"
        df = df.groupby("delta_loc").agg({"score": ["mean", "std"], "threshold": "mean", "new_threshold": "mean", "cluster_1_A_purity": ["mean", "std"], "cluster_1_B_purity": ["mean", "std"], "cluster_1_count": ["mean", "std"], "cluster_2_count": ["mean", "std"]})
        df = df.sort_values(by=[(plot_name,'mean')])
        plt.plot(df[plot_name].iloc[0], df["score"]["mean"].iloc[0],'X',markersize=12,color=color_list[color_index])
        plt.ylabel("Score", fontsize=20)
        plt.xlabel("Isomorphic threshold", fontsize=20)

    if pattern==10:
        plot_name = "new_threshold"
        # delta_loc が同じものをまとめて resultを積算する
        df = df.groupby("delta_loc").agg({"result": ["sum", "count"], "threshold": "mean", "new_threshold": "mean", "cluster_1_A_purity": ["mean", "std"], "cluster_1_B_purity": ["mean", "std"], "cluster_1_count": ["mean", "std"], "cluster_2_count": ["mean", "std"]})
        # resultの積算をnew_thresholdの値でソートする
        df = df.sort_values(by=[(plot_name,'mean')])
        # 積算したresultをnew_thresholdに対してプロットする
        plt.plot(df[plot_name].iloc[0], df['result']["sum"].iloc[0], 'x', color=color_list[color_index], markersize=15)
        plt.plot(df[plot_name], df["result"]["sum"],'o-',markersize=10,color=color_list[color_index],label=label_name)
        plt.ylabel("Success count", fontsize=20)
        plt.xlabel("Isomorphic threshold", fontsize=20)
        #plt.show()

    if pattern==11:
        plot_name = "new_threshold"
        # delta_loc が同じものをまとめて total_scoreの積算値と平均値を計算する
        df = df.groupby("delta_loc").agg({"total_score": ["sum", "mean"], "threshold": "mean", "new_threshold": "mean", "cluster_1_A_purity": ["mean", "std"], "cluster_1_B_purity": ["mean", "std"], "cluster_1_count": ["mean", "std"], "cluster_2_count": ["mean", "std"]})
        # resultの積算をnew_thresholdの値でソートする
        df = df.sort_values(by=[(plot_name,'mean')])
        # 平均したtotal_scoreをnew_thresholdに対してプロットする
        plt.plot(df[plot_name].iloc[0], df['total_score']["mean"].iloc[0], 'x', color=color_list[color_index], markersize=15)
        plt.plot(df[plot_name], df["total_score"]["mean"],'o-',markersize=5,linewidth=2.0,color=color_list[color_index],label=label_name)
        plt.ylabel("mean(total score)", fontsize=20)
        plt.xlabel("Isomorphic threshold", fontsize=20)

    # X tics 18
    plt.xticks(fontsize=18)
    # Y ticcs 18
    plt.yticks(fontsize=18)
    # プロットの上下左右を調整する
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    # 凡例を表示する
    plt.legend(loc='best', fontsize=18)
    plt.xlim(0,1.0)
    plt.hlines(0.9, 0,10,color='g', linestyles='dotted')

    color_index += 1

plt.savefig(f"score_{pattern}.png")
plt.show()

