import numpy as np
from scipy.stats import norm

def GaussianKernel(kernel, sigma):
    """Returns a 2D Gaussian kernel."""

    lim = kernel // 2 + (kernel % 2) / 2
    x = np.linspace(-lim, lim, kernel + 1)
    kern1d = np.diff(norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()

def DtGfiltersBank(sigma):
    x = np.arange(-5*sigma, 5*sigma + 1)
    G = GaussianKernel(len(x), sigma)
    Gx, Gy  = np.gradient(G)
        
    #Compute the 2,0, 1,1 and 0,2 order kernels
    Gxx, Gxy = np.gradient(Gx)
    Gyx, Gyy = np.gradient(Gy)
     
    kernels = np.append([G], [Gy.transpose()], 0)
    kernels = np.append(kernels, [Gx.transpose()], 0)
    kernels = np.append(kernels, [Gyy.transpose()], 0)
    kernels = np.append(kernels, [Gxy.transpose()], 0)
    kernels = np.append(kernels, [Gxx.transpose()], 0)
    return kernels