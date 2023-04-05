import sys
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
csv_file = sys.argv[1]
data = pd.read_csv(csv_file)

# グラフの大きさを設定
fig, ax = plt.subplots(figsize=(10, 6))

# 同じ loc_AB ごとに各カラムの平均値と標準偏差を取得
mean_data = data.groupby("loc_AB").mean().reset_index()
std_data = data.groupby("loc_AB").std().reset_index()

# 平均値と標準偏差を新しい DataFrame に結合
result_data = pd.concat([mean_data, std_data.drop("loc_AB", axis=1)], axis=1, keys=["mean", "std"])

# mode_AA, mode_AB, mode_BB のカラムをプロット
ax.plot(result_data["loc_AB"], result_data["mode_AA"], 'o', label="mode_AA")
ax.plot(result_data["loc_AB"], result_data["mode_AB"], 'o', label="mode_AB")
ax.plot(result_data["loc_AB"], result_data["mode_BB"], 'o', label="mode_BB")

print(result_data)

# グラフの設定
ax.set_xlabel("loc_AB")
ax.set_ylabel("mode")
ax.set_title("Mode distribution for each loc_AB")
ax.legend()

# グラフの表示
plt.show()
