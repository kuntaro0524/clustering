import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def custom_pdf(x, mu, sigma, a, b, c):
    baseline = a * x + b
    gauss = np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return c * gauss + baseline

x = np.linspace(-5, 5, 100)
y_true = custom_pdf(x, mu=0, sigma=1, a=0.2, b=0.5, c=1)

# ノイズを加えたデータを作成
np.random.seed(123)
y_data = y_true + np.random.normal(loc=0, scale=0.1, size=len(x))

# フィッティング
p0 = [0, 1, 1, 1, 1]  # 初期値
popt, pcov = curve_fit(custom_pdf, x, y_data, p0=p0)

# フィッティング結果の確認
print("mu: ", popt[0])
print("sigma: ", popt[1])
print("a: ", popt[2])
print("b: ", popt[3])
print("c: ", popt[4])

y_fit = custom_pdf(x, *popt)

plt.plot(x, y_true, label='True')
plt.plot(x, y_data, '.', label='Data')
plt.plot(x, y_fit, label='Fit')
plt.legend()
plt.show()
