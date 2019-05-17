import numpy as np
from computeOBIFs import computeOBIFs

def computeFeature(path):
    obifs_1 = computeOBIFs(path, sigma = 0.5, epsilon=1e-05)
    obifs_2 = computeOBIFs(path, sigma = 1, epsilon=1e-03)
    obifs_3 = computeOBIFs(path, sigma = 2, epsilon=1e-01)
    
    bins = list(np.arange(1, 31, 1))
    
    hist_1, _ = np.histogram(np.ravel(obifs_1), bins=bins)
    hist_1[np.argmax(hist_1)] = 0
    hist_2, _ = np.histogram(np.ravel(obifs_2), bins=bins)
    hist_2[np.argmax(hist_2)] = 0
    hist_3, _ = np.histogram(np.ravel(obifs_3), bins=bins)
    hist_3[np.argmax(hist_3)] = 0
    
    hist = hist_1 + hist_2 + hist_3
    return hist / np.max(hist)