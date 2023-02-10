import numpy as np
from scipy.stats import skewnorm
from matplotlib import pyplot as plt

def generate_random_from_skew_gaussian(alpha, loc, scale):
    return skewnorm.rvs(alpha, loc, scale)

alpha = -10 # skewness parameter
loc = 0.97 # location parameter
scale = 0.05 # scale parameter

random_numbers = [generate_random_from_skew_gaussian(alpha, loc, scale) for i in range(1000)]

alpha = -10 # skewness parameter
loc = 0.98 # location parameter
scale = 0.05 # scale parameter

random_numbers2 = [generate_random_from_skew_gaussian(alpha, loc, scale) for i in range(1000)]

plt.hist(random_numbers,alpha=0.5)
plt.hist(random_numbers2,alpha=0.5)
plt.show()
