import pandas as pd
import matplotlib.pyplot as plt
import sys

# CSVファイルを読み込む
df = pd.read_csv(sys.argv[1], header=None, names=["# of datasets", "Threshold"])

# データをデータ数ごとにグループ化して平均を計算
df_mean = df.groupby("DataSize").mean()
df_std = df.groupby("DataSize").std()

# プロット
fig, ax = plt.subplots()

ax.errorbar(df_mean.index, df_mean["Threshold"], yerr=df_std["Threshold"], fmt="o", capsize=3)

ax.set_xlabel("Data Size")
ax.set_ylabel("Threshold")

plt.show()
