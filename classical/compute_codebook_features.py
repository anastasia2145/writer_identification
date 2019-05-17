import os
import cv2
import numpy as np

import SegmentWords
import SegmentFeatures
from scipy.spatial.distance import euclidean

def dist(vector, features):
    distances = []
    for feature in features:
        distances.append(euclidean(vector, feature))
    distances = np.array(distances)
    return np.argmin(distances)
    
    
def transform(str_numb):
    l, r = str_numb.split("_")
    return float(l + "." + r)
    
def get_str2numb_numb2dict(vect):
    str_to_ind_dict = {}
    count = 0
    for v in vect:
        if v not in str_to_ind_dict.keys():
            str_to_ind_dict[v] = count
            count += 1
    reverse_dict = {v:k for k, v in str_to_ind_dict.items()}
    return str_to_ind_dict, reverse_dict

def apply_dict(dict_keys, X):
    res = []
    for x in X:
        res.append(dict_keys[x])
    return res
    
def compute_features(train_path, save_name_x, save_name_y):
    train_features = []
    train_y = []
    for variant in os.listdir(train_path):
        v = variant[-1]
        autors_path = os.path.join(train_path, variant)
        autors = os.listdir(autors_path)
        for autor in autors:
            train_y.append(str(v) + "_" + str(autor))
            words = os.listdir(os.path.join(autors_path, autor))
            path = os.path.join(autors_path, autor)
            autor_feature =  np.zeros((len(cluster_centres)))
            for word in words:
                word_path = os.path.join(path, word)
                image = cv2.imread(word_path)
                segments = SegmentWords.get_segments(image, kernel_size = 11)
                for segment in segments:
                    if np.sum(segment == 0) > 20  and (255 in segment):
                        feature = SegmentFeatures.get_features(np.stack((segment,)*3, axis=-1))
                        autor_feature[dist(feature, cluster_centres)] += 1
            train_features.append(autor_feature / max(autor_feature))
            
    np.save(save_name_x, train_features)
    np.save(save_name_y, train_y)
    return train_features, train_y
    
    