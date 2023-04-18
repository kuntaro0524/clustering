import numpy as np
import matplotlib.pyplot as plt

def custom_pdf(x):
    mu = 0
    sigma = 1
    a = 0.1
    b = 0.1
    c = 0.1
    baseline = a * x + b
    gauss = np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return c * gauss + baseline

x = np.linspace(-5, 5, 100)
pdf = custom_pdf(x)

plt.plot(x, pdf)

samples = np.random.normal(loc=0, scale=1, size=1000)
prob = custom_pdf(samples)

# probをプロットします
plt.hist(samples, bins=100, density=True)
plt.plot(x, pdf)
plt.show()

#  probに対して、逆にcustom_pdfの係数を求めたい
#  ここにコードを書いてください
#  ここまで