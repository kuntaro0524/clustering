import pandas as pd
import numpy as np
import sys

# header ないCSV
# 1列目がnum_datasets、2列目がthresholdの数値
df = pd.read_csv(sys.argv[1], header=None)
df.columns = ["num_datasets", "threshold"]

# thresholdの平均値を計算して出力する
# thresholdの標準偏差を計算して出力する
print(df["threshold"].mean())
print(df["threshold"].std())

