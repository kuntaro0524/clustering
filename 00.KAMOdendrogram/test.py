import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]

fig, ax = plt.subplots()
ax.plot(x, y)

ax.annotate("Comment here", (2, 4), xycoords='data',
            xytext=(-150, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2"))

plt.show()