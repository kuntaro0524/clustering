# ヒストグラムをプロットする

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import sys
import pandas as pd

# CCのCSVファイルを読み込む。１行に１個CCの値がある
df = pd.read_csv(sys.argv[1])
# CCの値を取り出す
cc = df['cc']

# ヒストグラムを作成する
# binsはヒストグラムの棒の数
# rangeはヒストグラムの範囲
# 図を作成する

plt.hist(cc, bins=10, range=(0,1),density=False)
plt.xlabel('CC')
plt.ylabel('Frequency')
# 図の範囲は(0.8,1)
#plt.xlim(0.8,1)
plt.title('CC histogram')
plt.grid(True)
plt.show()


