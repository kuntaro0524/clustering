import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
print(df)

# scatter plot
import matplotlib.pyplot as plt
plt.title("Ward distance and new threshold for each scale")
plt.xlabel("Ward distance")
# ward_distanceに対して、c1_A_purity をプロットする
# ward_distanceに対して、c1_B_purity をプロットする
#plt.scatter(df['ward_distance'], df['c1_A_purity'], label="c1_A_purity")
#plt.scatter(df['ward_distance'], df['c1_B_purity'], label="c1_B_purity")

plt.scatter(df['new_thresh'], df['c1_A_purity'], label="c1_A_purity")
plt.scatter(df['new_thresh'], df['c1_B_purity'], label="c1_B_purity")

plt.show()

