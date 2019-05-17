import numpy as np
from computeBIFs import computeBIFs


def atan2d(y, x):
    return np.arctan2(y, x) * 180 / np.pi
  

def quantization(a):
    directionAngles = [0, -45, -90, -135, -180, 180, 135, 90, 45]
    return np.argmin(np.abs(np.array(directionAngles) - a))

oBIFsQuantization = np.vectorize(quantization)
  
def computeOBIFs(img_path, sigma=0.5, epsilon=1e-05):   
    bifs, jet = computeBIFs(img_path, sigma, epsilon)
    
    obifs = np.zeros((bifs.shape))
    obifs[bifs == 1] = 1
    
    mask = bifs == 2
    
    slope_gradient = atan2d(jet[2], jet[1])
    slope_gradient = oBIFsQuantization(slope_gradient) + 1
    slope_gradient = np.array(slope_gradient, np.uint8)

    slope_gradient[slope_gradient == 6] = 5
    slope_gradient[slope_gradient > 5] -= 1
    
    obifs[mask] = 1 + slope_gradient[mask]

    gradient = np.arctan((2 * jet[4]) / (jet[5] - jet[4]))
    gradient = oBIFsQuantization(gradient) + 1
    gradient = np.array(gradient, np.uint8)
    
    gradient[gradient == 5] = 1
    gradient[gradient == 6] = 1
    gradient[gradient == 7] = 2
    gradient[gradient == 8] = 3
    gradient[gradient == 9] = 4

    mask = bifs == 3
    obifs[mask] = 10

    mask = bifs == 4
    obifs[mask] = 11

    mask = bifs == 5
    obifs[mask] = 11 + gradient[mask]

    mask = bifs == 6
    obifs[mask] = 15 + gradient[mask]

    mask = bifs == 7
    obifs[mask] = 19 + gradient[mask]

    return obifs