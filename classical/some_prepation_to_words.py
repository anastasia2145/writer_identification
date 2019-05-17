import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def filter_word(image):
    img = image[:,:,0]
    hist = np.sum(1 - img/255, axis=1)
    indices_black = np.ravel(np.argwhere(hist > 0))
    start = indices_black[0]
    black_areas = []
    for i, ind in enumerate(indices_black[:-1]):
        if ind != indices_black[i + 1] - 1:
            black_areas.append((start, ind))
            start = indices_black[i + 1]
        if indices_black[i + 1] == indices_black[-1]:
            black_areas.append((start, indices_black[i + 1]))
    if len(black_areas) == 0:
        return image
    
    max_length = -1
    word_area = ()
    for area in black_areas:
        if (area[1] - area[0]) > max_length:
            max_length = area[1] - area[0]
            word_area = area
    n, m, _ = image.shape
    start = 0 if (word_area[0] - 5 < 0 ) else word_area[0] - 5
    end = n if (word_area[1] + 5 >= n ) else word_area[1] + 5
    return image[start:end, :]
    
    
def move_in_one_folder(path, save_path, csv_name):
    names = []
    autor_inds = []
    for variant in os.listdir(path):
        autor_path = os.path.join(path, variant)
        for autor in os.listdir(autor_path):
            words_path = os.path.join(autor_path, autor)
            for word in os.listdir(words_path):
                img = cv2.imread(os.path.join(words_path, word))
                new_name = str(variant[-1]) + '_' + str(autor) + '_' + word
                if np.sum(1 - (img[:,:, 0] /255)) > 150:
                    img = filter_word(img)
                    cv2.imwrite(os.path.join(save_path, new_name), img)
                    names.append(new_name)
                    autor_inds.append(str(variant[-1]) + '_' + str(autor))
    pd.DataFrame({"file_name": names, "label": autor_inds}) \
        .to_csv(csv_name, index=False, header=True, columns = ["file_name", "label"])    
        
        
def compute_max_shape(path):
    height, width = 0, 0 
    for word in os.listdir(path):
        word_path = os.path.join(path, word)
        img = cv2.imread(word_path)
        n, m, _ = img.shape
        if n > height:
            height = n
        if m >  width:
             width = m
    return (height, width)