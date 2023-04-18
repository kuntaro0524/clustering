import pandas as pd
import matplotlib.pyplot as plt
import sys

# CSVファイルを読み込む
df1 = pd.read_csv(sys.argv[1], header=None, names=["DataSize", "Threshold"])
df2 = pd.read_csv(sys.argv[2], header=None, names=["DataSize", "Threshold"])

# データをデータ数ごとにグループ化して平均を計算
df1_mean = df1.groupby("DataSize").mean()
df1_std = df1.groupby("DataSize").std()
df2_mean = df2.groupby("DataSize").mean()
df2_std = df2.groupby("DataSize").std()

# プロット
fig, ax = plt.subplots()

# 線も引く
ax.plot(df1_mean.index, df1_mean["Threshold"], label="Trypsin")
ax.plot(df2_mean.index, df2_mean["Threshold"], label="Trn")

ax.errorbar(df1_mean.index, df1_mean["Threshold"], yerr=df1_std["Threshold"], fmt="o", capsize=3)
ax.errorbar(df2_mean.index, df2_mean["Threshold"], yerr=df2_std["Threshold"], fmt="o", capsize=3)

ax.set_xlabel("Data Size")
ax.set_ylabel("Threshold")
#　凡例を表示
ax.legend()

# y=0.6 y=0.7 に囲まれた領域を薄いピンク色にする
ax.axhspan(0.6, 0.7, facecolor="pink", alpha=0.5)

plt.savefig("paper.png")
plt.show()

# データフレームをマージする(df1_mean, df2_mean, df1_std, df2_std)
# データフレームの名前をつけて管理する
newdf = pd.DataFrame()
newdf["Trypsin"] = df1_mean["Threshold"]
newdf["Trn"] = df2_mean["Threshold"]
newdf["Trypsin_std"] = df1_std["Threshold"]
newdf["Trn_std"] = df2_std["Threshold"] 


# csvとして出力
newdf.to_csv("paper.csv")
