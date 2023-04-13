import numpy as np
import matplotlib.pyplot as plt

# 表示するデータの作成
x = np.arange(-np.pi, np.pi, 0.1)

y1 = np.cos(x * 1)
y2 = np.cos(x * 2)
y3 = np.cos(x * 3)

# グラフ領域の作成
fig = plt.figure(figsize = [5.8, 4])

# 座標軸の作成
ax1 = fig.add_subplot(2, 3, 1) # ２行３列の１番目
ax2 = fig.add_subplot(2, 3, 2) # ２行３列の２番目
ax3 = fig.add_subplot(2, 3, 3) # ２行３列の３番目
ax4 = fig.add_subplot(2, 1, 2) # ２行１列の２番目

# データのプロット
ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y1+y2+y3)

# グラフの表示
plt.show()
