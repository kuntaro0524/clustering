import numpy as np

# rv_continuousをインポート
from scipy.stats import rv_continuous
class MyDistribution(rv_continuous):
    def __init__(self, spline, a, b):
        self.spline = spline
        x_range = np.linspace(a,b, num=1000)
        self.pdf = self.spline(x_range)
        self.pdf = self.pdf / np.sum(self.pdf)

        super().__init__(a=a, b=b)

    def _pdf(self, x):
        return self.spline(x) / np.sum(self.pdf)

    def sample(self, size=None):
        return self.rvs(size=size)

class CCmodel:
    def __init__(self, cc_values):
        self.cc_values = cc_values
        # cc_values の頻度分布を hist, bin_edges に格納する
        hist, bin_edges = np.histogram(cc_values, bins=100, density=True)
        # bin_edges の中心値を bin_centers に格納する
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Akima1DInterpolatorをインポート
        from scipy.interpolate import Akima1DInterpolator
        self.spline = Akima1DInterpolator(bin_centers, hist)

        self.x_range = np.linspace(bin_centers[0], bin_centers[-1], num=1000)
        pdf = self.spline(self.x_range)
        self.pdf = pdf / np.sum(pdf)

        self.mydist = MyDistribution(self.spline, a=bin_centers[0], b=bin_centers[-1])
         
    def rand_func(self,i):
        return self.mydist.rvs()
    
    def rand_fast(self,i):
        #value = self.mydist.sample(1)
        value = self.mydist._rvs()
        print(value)
        return value

    def calcCCmulti(self, ndata, nproc=24):
        import multiprocessing as mp
        with mp.Pool(nproc) as p:
            cc_list = p.map(self.rand_func, [None] * ndata)
        cc_array = np.array(cc_list)

        return cc_array

    def plotFunction(self, figname,xstart=0,xend=1.0):
        import matplotlib.pyplot as plt
        x_range = np.linspace(xstart,xend, num=1000)
        plt.plot(x_range, self.pdf)
        plt.savefig(figname)
        plt.close()

# main program
if __name__ == '__main__':
    ccmodel=CCmodel(np.random.normal(0,1,1000))
    cc_array = ccmodel.calcCCmulti(1000)
    print(cc_array)
    # cc_array の頻度分布を表示
    import matplotlib.pyplot as plt
    plt.hist(cc_array, bins=10, density=False)
    plt.show()

    # cc_array をファイルに書き出す
    np.savetxt('cc_array.txt', cc_array)
    