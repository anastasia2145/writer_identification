from computeOBIFs import computeOBIFs
import os
import cv2
import numpy as np
import matplotlib.pylab as plt
from colorBIFs import bifs_to_color_image
from collections import Counter
from computeFeature import computeFeature


def show(img):
    """
    show rgb image
    """
    ax = plt.axes([0,0,4,4], frameon=False)
    ax.set_axis_off()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
def compute_features(path):
    features = []
    print('-----------------------------')
    for image_path in os.listdir(path):
        features.append(computeFeature(os.path.join(path, image_path)))
        print(image_path)
    return np.array(features)
        
if __name__ == "__main__":   
    # train_path = 'C:/Users/Anastasia/Pictures/train_areas'
    # val_path = 'C:/Users/Anastasia/Pictures/val_areas'
    # test_path = 'C:/Users/Anastasia/Pictures/test_areas'
    
    # train_features = compute_features(train_path)
    # val_features = compute_features(val_path)
    # test_features = compute_features(test_path)
    
    # np.save('train_obifs', train_features)
    # np.save('val_obifs', val_features)
    # np.save('test_obifs', test_features)
    
    path = 'C:/Users/Anastasia/Documents/Some_shit/obif_test.png'
    img = cv2.imread(path)
    print(img)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # obifs_1 = computeOBIFs(path, sigma = 0.5, epsilon=1e-05)
    # obifs_2 = computeOBIFs(path, sigma = 1, epsilon=1e-09)
    # obifs_3 = computeOBIFs(path, sigma = 2, epsilon=1e-018)
    obifs_1 = computeOBIFs(path, sigma = 0.5, epsilon=1e-05)
    obifs_2 = computeOBIFs(path, sigma = 1, epsilon=1e-09)
    obifs_3 = computeOBIFs(path, sigma = 2, epsilon=1e-015)
    
    # print(obifs_1[260:270, 30:40])
    # print(np.min(obifs_1))
    # print(obifs_2[260:270, 30:40])
    # print(np.min(obifs_2))
    # print(obifs_3[260:270, 30:40])
    # print(np.min(obifs_3))
    
    bins = list(np.arange(1, 31, 1))
    hist_1, _ = np.histogram(np.ravel(obifs_1), bins=bins)
    hist_1[0] = 0
    hist_2, _ = np.histogram(np.ravel(obifs_2), bins=bins)
    hist_2[0] = 0
    hist_3, _ = np.histogram(np.ravel(obifs_3), bins=bins)
    hist_3[0] = 0
    
    hist = hist_1 + hist_2 + hist_3
    print(hist)
    obifs_1 = bifs_to_color_image(obifs_1)
    obifs_2 = bifs_to_color_image(obifs_2)
    obifs_3 = bifs_to_color_image(obifs_3)
    print(img.shape)
    print(obifs_1.shape)
    print(obifs_2.shape)
    print(obifs_3.shape)
    img = np.vstack((img, np.vstack((obifs_1, np.vstack((obifs_2, obifs_3)) )) ))
    cv2.imwrite('test_results_3.png', img)
    print(img.shape)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # show(img)