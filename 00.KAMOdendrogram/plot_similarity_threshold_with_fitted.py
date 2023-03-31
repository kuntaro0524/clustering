import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# CSVファイルのリストを取得
csv_files = glob("*.csv")

# グラフのスタイルを設定
#plt.style.use("ggplot")
#plt.style.use("seaborn")

labels=["simulation 4","simulation 5"]
plt.rcParams.update({"font.size": 18})  # フォントサイズを14に設定

fig, ax = plt.subplots(figsize=(10, 6))  # 横幅10インチ、縦幅6インチの図を作成

# 各CSVファイルに対して処理を実行
for i, csv_file in enumerate(csv_files):
    # データの読み込み
    data = pd.read_csv(csv_file)

    # エラーバー付きの散布図をプロット
    this_label = labels[i]
    plt.errorbar(data["n_data"], data["threshold"], yerr=data["sigma"], fmt="o", label=f"{this_label}")

    # 多項式フィッティング
    degree = 2  # 2次多項式でフィッティング
    coeffs = np.polyfit(data["n_data"], data["threshold"], degree)
    poly = np.poly1d(coeffs)
    
    # フィッティングされたカーブをプロット
    x = np.linspace(min(data["n_data"]), max(data["n_data"]), 100)
    y = poly(x)
    #plt.plot(x, y, label=f"Fitted Curve {i+1}",alpha=0.2)
    plt.plot(x, y, alpha=0.1, linewidth=5.0)

# グラフの設定
plt.xlabel("# of datasets")
plt.ylabel("mean similarity threshold")
plt.legend()

# グラフの表示
plt.savefig("simulation4-5.png")
plt.show()