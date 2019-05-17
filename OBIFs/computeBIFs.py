import cv2
import numpy as np
from scipy import ndimage
from DtGfiltersBank import DtGfiltersBank

def computeBIFs(im, sigma, epsilon):
    gray = cv2.imread(im)
    gray = gray[:, :, 0] / 255
    
    # Dervative orders list
    orders = [0,1,1,2,2,2]

    # Set jets arrays
    jet = np.zeros((6, gray.shape[0], gray.shape[1]), np.float64)

    # Do the actual computation
    DtGfilters = DtGfiltersBank(sigma)

    for i in range(0, 6):
        jet[i] = ndimage.filters.convolve(gray, DtGfilters[i], mode='constant')*(sigma**orders[i])

    # Compute lambda and mu
    _lambda = 0.5*(np.squeeze(jet[3]) + np.squeeze(jet[5]))
    mu = np.sqrt(0.25 * ((np.squeeze(jet[3]) - np.squeeze(jet[5])) ** 2) + np.squeeze(jet[4]) ** 2)

    # Initialize classifiers array
    c = np.zeros((jet.shape[1], jet.shape[2], 7), np.float64)

    # Compute classifiers
    c[:, :, 0] = epsilon * np.squeeze(jet[0])
    c[:, :, 1] = np.sqrt(np.squeeze(jet[1]) ** 2 + np.squeeze(jet[2]) ** 2)
    c[:, :, 2] = _lambda
    c[:, :, 3] = -_lambda
    c[:, :, 4] = 2 ** (-1 / 2.0) * (mu + _lambda)
    c[:, :, 5] = 2 ** (-1 / 2.0) * (mu - _lambda)
    c[:, :, 6] = mu

    return [np.array(c[:, :].argmax(2)+1, dtype=np.uint8), jet]
    
    
    
def computeBIFs_(img_path, sigma, epsilon, configuration):
    img = cv2.imread(img_path)
    img = img[:,:,0] / 255
    print(img.shape)
    
    orders = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
    
    n, m = img.shape
    jet = []
    
    DtGfilters = DtGfiltersBank(sigma, configuration)
    
    for i in range(len(orders)):
#         jet[i,:,:] = efficientConvolution(img, DtGfilters[i][0],DtGfilters[i][1])*(sigma ** sum(orders[i,:]))
        jet.append(efficientConvolution(img, DtGfilters[i][0],DtGfilters[i][1])*(sigma ** sum(orders[i,:])))
    
    jet = np.array(jet)
    if configuration==1:
        # Compute lambda and mu
        lambda_ = np.squeeze(jet[3,:,:]) + np.squeeze(jet[5,:,:])
        mu = (((np.squeeze(jet[3,:,:]) - np.squeeze(jet[5,:,:])) ** 2) + 4 * (np.squeeze(jet[4,:,:]) ** 2)) ** 0.5      
    else:
        # Compute lambda and mu
        lambda_ = 0.5 * (np.squeeze(jet[3,:,:]) + np.squeeze(jet[5,:,:]))
        mu = (0.25 * (((np.squeeze(jet[3,:,:]) - np.squeeze(jet[5,:,:])) ** 2) + (np.squeeze(jet[4,:,:]) ** 2))) ** 0.5

    c = np.zeros((jet.shape[1], jet.shape[2], 7), np.float64)

    # Compute classifiers
    c[:, :, 0] = epsilon * np.squeeze(jet[0])
    c[:, :, 1] = np.sqrt(np.squeeze(jet[1]) ** 2 + np.squeeze(jet[2]) ** 2)
    c[:, :, 2] = lambda_
    c[:, :, 3] = -lambda_
    c[:, :, 4] = 2 ** (-1 / 2.0) * (mu + lambda_)
    c[:, :, 5] = 2 ** (-1 / 2.0) * (mu - lambda_)
    c[:, :, 6] = mu
    
#     c = np.array(c)
    bifs = np.argmax(c, axis=2)
    return bifs, jet