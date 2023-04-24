import pandas as pd
# thresh.datを読み込む
# 内容はCSV

pd = pd.read_csv("thresh.dat", header=None, names=["scale", "ward_dist", "new_thresh"])

# このデータをプロットする
# 横軸はscale
# 縦軸は ward_dist, new_thresh
# それぞれのプロットにはラベルをつける
# ward_dist は "Ward distance"
# new_thresh は "New threshold"
# また、プロットのタイトルは "Ward distance and new threshold for each scale"
# とする

# ポイントも描く
import matplotlib.pyplot as plt
 
plt.title("Ward distance and new threshold for each scale")
plt.xlabel("Scale")
plt.ylabel("Ward distance and new threshold")
plt.plot(pd["scale"], pd["ward_dist"], 'o-',label="Ward distance")

plt.plot(pd["scale"], pd["new_thresh"], 'o-',label="New threshold")
plt.show()


