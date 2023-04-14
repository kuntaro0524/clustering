from scipy.stats import rv_continuous, lognorm

class inv_lognorm(rv_continuous):
    def _pdf(self, x, sigma, shift, scale):
        return lognorm.pdf(1 - x, sigma, shift, scale) / (1 - x)


inv_ln = inv_lognorm(a=0, b=1, name='inv_lognorm')
rvs = inv_ln.rvs(sigma, shift, scale, size=1000)


